"""Run the synthetic RAG benchmark, restoring the live knowledge base afterward."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis.asyncio as redis

from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.keyword_index import RedisKeywordIndex
from app.services.milvus_store import get_milvus_store
from evals.run_rag_eval import _request_json, _parse_top_ks, evaluate, load_cases


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_review_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id", "case_type", "question", "expected_answer", "answer_text",
        "relevant_ids", "retrieved_ids", "citation_ids", "answer_correct_0_or_1",
        "grounded_0_or_1", "citation_correct_0_or_1", "abstained_0_or_1", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "case_type": row["case_type"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "answer_text": row.get("answer_text", ""),
                "relevant_ids": json.dumps(row["relevant_ids"], ensure_ascii=False),
                "retrieved_ids": json.dumps(row["retrieved_ids"], ensure_ascii=False),
                "citation_ids": json.dumps(row.get("citation_ids", []), ensure_ascii=False),
            })


async def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus = _read_jsonl(Path(args.corpus))
    cases = load_cases(Path(args.cases))
    ids = {str(doc.get("id")) for doc in corpus}
    if not corpus or len(ids) != len(corpus) or not all(item.startswith("demo-rag-") for item in ids):
        raise ValueError("Demo corpus must contain unique demo-rag-* document IDs")
    unknown = {source for case in cases for source in case.relevant_ids if source not in ids}
    if unknown:
        raise ValueError(f"Test cases refer to missing demo documents: {sorted(unknown)}")

    health = _request_json("GET", f"{args.base_url.rstrip('/')}/api/v1/health", timeout_s=8)
    if health.get("status") != "ok":
        raise RuntimeError("The application health endpoint is not ok")

    store = get_milvus_store()
    if not store.connect():
        raise RuntimeError("Milvus is unavailable")
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    index = RedisKeywordIndex(redis_client)
    inserted: list[str] = []
    try:
        if store.list_documents(limit=1) or await redis_client.hlen(settings.keyword_index_redis_key):
            raise RuntimeError("Demo evaluation requires an empty knowledge base; no documents were changed")

        vectors = await EmbeddingService().embed_texts([str(doc["content"]) for doc in corpus])
        if any(len(vector) != settings.embedding_dimensions for vector in vectors):
            raise ValueError("Embedding dimension does not match Milvus collection")
        for doc, vector in zip(corpus, vectors, strict=True):
            doc_id = str(doc["id"])
            store.insert_vector(doc_id, str(doc["title"]), str(doc["doc_type"]), str(doc["content"]), vector)
            inserted.append(doc_id)
        await index.upsert_many({
            "chunk_id": str(doc["id"]),
            "parent_doc_id": str(doc["id"]),
            "title": str(doc["title"]),
            "doc_type": str(doc["doc_type"]),
            "text": str(doc["content"]),
            "metadata": {"evaluation_only": True},
        } for doc in corpus)

        retrieval = evaluate(args.base_url.rstrip("/"), cases, top_ks=args.top_ks, include_chat=False)
        chat_rows: list[dict[str, Any]] = []
        if args.chat_sample:
            positive = [case for case in cases if case.answerable]
            negative = [case for case in cases if not case.answerable]
            selected = positive[:max(0, args.chat_sample - int(bool(negative)))]
            if negative:
                selected += negative[:1]
            chat_rows = evaluate(
                args.base_url.rstrip("/"), selected,
                top_ks=[max(args.top_ks)], include_chat=True,
            )["cases"]

        report = {
            "dataset_kind": "synthetic_demo_not_business_validation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_count": len(corpus),
            "models": {
                "embedding": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                "rerank": settings.rag_reranker_model if settings.rag_reranker_enabled else None,
                "chat": settings.openai_model,
            },
            "retrieval": retrieval,
            "chat_sample": chat_rows,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if chat_rows:
            _write_review_sheet(output.with_suffix(".review.csv"), chat_rows)
        return report
    finally:
        try:
            for doc_id in inserted:
                await index.delete(doc_id=doc_id)
            remaining = set(inserted)
            for attempt in range(6):
                for doc_id in remaining:
                    store.delete_document(doc_id=doc_id)
                remaining = {
                    str(doc["id"])
                    for doc in store.list_documents(limit=500)
                    if str(doc.get("id")) in inserted
                }
                if not remaining:
                    break
                await asyncio.sleep(0.5 * (attempt + 1))
            if remaining:
                raise RuntimeError(f"Demo cleanup could not remove: {sorted(remaining)}")
        finally:
            await redis_client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus", default="evals/demo_corpus.jsonl")
    parser.add_argument("--cases", default="evals/demo_cases.jsonl")
    parser.add_argument("--top-ks", default="1,3,5")
    parser.add_argument("--chat-sample", type=int, default=0)
    parser.add_argument("--output", default="evals/runs/demo-report.json")
    args = parser.parse_args()
    args.top_ks = _parse_top_ks(args.top_ks, 5)
    if max(args.top_ks) > 5 or args.chat_sample < 0:
        parser.error("top-ks must be at most 5 and chat-sample must be nonnegative")
    report = asyncio.run(run(args))
    retrieval = report["retrieval"]
    print(json.dumps({
        "status": "ok",
        "dataset_kind": report["dataset_kind"],
        "case_count": retrieval["case_count"],
        "hit@5": retrieval.get("hit@5"),
        "mrr": retrieval["mrr"],
        "report": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
