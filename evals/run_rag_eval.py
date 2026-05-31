from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.request import ProxyHandler, build_opener


_OPENER = build_opener(ProxyHandler({}))


@dataclass
class RagCase:
    case_id: str
    question: str
    relevant_ids: list[str]
    keywords: list[str]

    @property
    def expected_doc_ids(self) -> list[str]:
        return self.relevant_ids


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with _OPENER.open(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_cases(path: Path) -> list[RagCase]:
    cases: list[RagCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        cases.append(
            RagCase(
                case_id=str(item.get("id") or f"case-{line_no}"),
                question=str(item["question"]),
                relevant_ids=[
                    str(x)
                    for x in (item.get("relevant_ids") or item.get("expected_doc_ids") or [])
                ],
                keywords=[str(x) for x in item.get("keywords", [])],
            )
        )
    return cases


def _doc_identities(item: dict[str, Any]) -> list[str]:
    values = [
        item.get("id"),
        item.get("doc_id"),
        item.get("title"),
        item.get("content"),
    ]
    return [str(value) for value in values if value is not None]


def _matches_expected(item: dict[str, Any], expected_doc_ids: list[str]) -> bool:
    identities = _doc_identities(item)
    return any(
        expected and any(expected in identity for identity in identities)
        for expected in expected_doc_ids
    )


def _find_rank(results: list[dict[str, Any]], expected_doc_ids: list[str]) -> int | None:
    expected = set(expected_doc_ids)
    for idx, item in enumerate(results, start=1):
        if _matches_expected(item, list(expected)):
            return idx
    return None


def _count_relevant(results: list[dict[str, Any]], relevant_ids: list[str], *, top_k: int) -> int:
    if not relevant_ids:
        return 0
    return sum(1 for item in results[:top_k] if _matches_expected(item, relevant_ids))


def _retrieval_metrics(
    results: list[dict[str, Any]],
    relevant_ids: list[str],
    *,
    top_k: int,
) -> dict[str, float | bool]:
    relevant_found = _count_relevant(results, relevant_ids, top_k=top_k)
    relevant_total = len(set(relevant_ids))
    return {
        f"hit@{top_k}": relevant_found > 0,
        f"precision@{top_k}": relevant_found / top_k if top_k else 0.0,
        f"recall@{top_k}": relevant_found / relevant_total if relevant_total else 0.0,
    }


def _parse_top_ks(value: str, fallback: int) -> list[int]:
    if not value.strip():
        return [fallback]
    out: list[int] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        top_k = int(text)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k not in out:
            out.append(top_k)
    return sorted(out or [fallback])


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]


def _answer_text(chat: dict[str, Any]) -> str:
    try:
        return str(chat["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


def _citation_hit(chat: dict[str, Any], expected_doc_ids: list[str]) -> bool:
    citations = chat.get("citations")
    if not isinstance(citations, list):
        return False
    expected = set(expected_doc_ids)
    for item in citations:
        if not isinstance(item, dict):
            continue
        if _matches_expected(item, list(expected)):
            return True
    return False


def evaluate(
    base_url: str,
    cases: list[RagCase],
    *,
    top_ks: list[int],
    include_chat: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    max_top_k = max(top_ks)
    latencies_ms: list[float] = []
    for case in cases:
        q = urlencode({"q": case.question, "top_k": str(max_top_k)})
        started = time.perf_counter()
        search = _request_json("GET", f"{base_url}/api/v1/documents/search?{q}")
        latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)
        results = search.get("results") if isinstance(search.get("results"), list) else []
        rank = _find_rank(results, case.relevant_ids)
        primary_k = max_top_k
        row: dict[str, Any] = {
            "id": case.case_id,
            "rank": rank,
            "hit": rank is not None and rank <= primary_k,
            "rr": 0.0 if rank is None else 1.0 / rank,
            "relevant_count": len(set(case.relevant_ids)),
            "retrieved_ids": [str(item.get("id")) for item in results[:max_top_k]],
            "retrieval_latency_ms": round(latency_ms, 2),
        }
        for top_k in top_ks:
            row.update(_retrieval_metrics(results, case.relevant_ids, top_k=top_k))

        if include_chat:
            chat = _request_json(
                "POST",
                f"{base_url}/api/v1/chat",
                payload={
                    "messages": [{"role": "user", "content": case.question}],
                    "stream": False,
                    "session_id": f"rag-eval-{case.case_id}-{int(time.time())}",
                },
                timeout_s=90.0,
            )
            answer = _answer_text(chat)
            row["keyword_ok"] = all(
                keyword.lower() in answer.lower() for keyword in case.keywords
            )
            row["citation_ok"] = _citation_hit(chat, case.relevant_ids)
            row["answer_preview"] = answer[:120]
        rows.append(row)

    total = len(rows) or 1
    report: dict[str, Any] = {
        "case_count": len(rows),
        "top_ks": top_ks,
        "mrr": sum(float(row["rr"]) for row in rows) / total,
        "retrieval_latency_avg_ms": sum(latencies_ms) / total,
        "retrieval_latency_p95_ms": _p95(latencies_ms),
        "cases": rows,
    }
    for top_k in top_ks:
        for metric in ("hit", "precision", "recall"):
            key = f"{metric}@{top_k}"
            report[key] = sum(float(row[key]) for row in rows) / total
    if include_chat:
        report["keyword_accuracy"] = sum(1 for row in rows if row["keyword_ok"]) / total
        report["citation_accuracy"] = sum(1 for row in rows if row["citation_ok"]) / total
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default="evals/rag_cases.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-ks", default="", help="Comma-separated K values, e.g. 1,3,5")
    parser.add_argument("--skip-chat", action="store_true", help="Only evaluate retrieval metrics")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    top_ks = _parse_top_ks(args.top_ks, args.top_k)
    try:
        health = _request_json("GET", f"{args.base_url}/api/v1/health", timeout_s=5.0)
    except URLError as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except TimeoutError as exc:
        print(json.dumps({"status": "timeout", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    checks = health.get("checks") if isinstance(health, dict) else {}
    if isinstance(checks, dict) and checks.get("milvus") is False:
        print(
            json.dumps(
                {
                    "status": "not_evaluable",
                    "reason": "Milvus is unavailable, so RAG retrieval cannot be evaluated.",
                    "health": health,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3

    try:
        report = evaluate(
            args.base_url.rstrip("/"),
            cases,
            top_ks=top_ks,
            include_chat=not args.skip_chat,
        )
    except Exception as exc:  # noqa: BLE001 - eval should report failures plainly
        print(
            json.dumps(
                {"status": "failed", "error": str(exc), "health": health},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 4

    print(json.dumps({"status": "ok", "health": health, "report": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
