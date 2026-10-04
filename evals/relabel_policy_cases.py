"""Bind policy questions to the currently indexed PDF chunks after rechunking."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import redis.asyncio as redis

from app.config import settings


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def relabel_cases(cases: list[dict[str, Any]], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        groups: dict[int, list[str]] = {}
        evidence: list[dict[str, Any]] = []
        for source in case.get("evidence", []):
            source_group = int(source["group"])
            groups.setdefault(source_group, [])
            for document in documents:
                metadata = document.get("metadata") or {}
                pages = metadata.get("page_numbers") or [metadata.get("page_number")]
                if source.get("page") not in pages:
                    continue
                if _compact(str(source["quote"])) not in _compact(str(document.get("text") or "")):
                    continue
                chunk_id = str(document["chunk_id"])
                if chunk_id not in groups[source_group]:
                    groups[source_group].append(chunk_id)
                evidence.append({**source, "chunk_id": chunk_id})
        if case.get("answerable") and (not groups or any(not group for group in groups.values())):
            missing = [index for index, group in groups.items() if not group]
            raise ValueError(f"{case['id']}: no indexed evidence for groups {missing}")
        relevance_groups = [groups[index] for index in sorted(groups)]
        relevant_ids = list(dict.fromkeys(doc_id for group in relevance_groups for doc_id in group))
        output.append({
            **case,
            "relevance_groups": relevance_groups,
            "relevant_ids": relevant_ids,
            "evidence": evidence,
            "chunking_strategy": "article_first_embedding_sentence_similarity",
        })
    return output


async def _load_indexed_documents(parent_doc_id: str) -> list[dict[str, Any]]:
    client = redis.from_url(settings.redis_url, decode_responses=True)
    try:
        raw = await client.hgetall(settings.keyword_index_redis_key)
    finally:
        await client.aclose()
    documents: list[dict[str, Any]] = []
    for value in raw.values():
        try:
            item = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("parent_doc_id") == parent_doc_id:
            documents.append(item)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/real_policy_cases.jsonl")
    parser.add_argument("--output", default="evals/runs/real-policy-cases-semantic.jsonl")
    args = parser.parse_args()
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("No cases to relabel")
    parent_ids = {case.get("source_parent_doc_id") for case in cases}
    if len(parent_ids) != 1:
        raise ValueError("Cases must refer to one parent document")
    documents = asyncio.run(_load_indexed_documents(str(next(iter(parent_ids)))))
    if not documents:
        raise RuntimeError("No indexed chunks found for source document")
    output = relabel_cases(cases, documents)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in output), encoding="utf-8")
    print(json.dumps({"cases": len(output), "indexed_chunks": len(documents), "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
