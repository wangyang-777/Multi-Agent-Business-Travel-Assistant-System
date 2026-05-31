from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import ProxyHandler, Request, build_opener


_OPENER = build_opener(ProxyHandler({}))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    with _OPENER.open(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_documents(base_url: str, *, limit: int) -> list[dict[str, Any]]:
    payload = _request_json("GET", f"{base_url.rstrip('/')}/api/v1/documents?limit={limit}")
    docs = payload.get("documents")
    return docs if isinstance(docs, list) else []


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/") + "/"
    if normalized.endswith("/chat/completions/"):
        return normalized
    return urljoin(normalized, "chat/completions")


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _call_llm(
    *,
    llm_base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = _request_json(
        "POST",
        _chat_completions_url(llm_base_url),
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


def _keywords_from_text(text: str, *, limit: int = 8) -> list[str]:
    values: list[str] = []
    patterns = [
        r"\d+(?:\.\d+)?\s*(?:CNY|元|天|晚)?",
        r"[A-Za-z][A-Za-z0-9_-]{1,24}",
        r"[\u4e00-\u9fa5]{2,8}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if value and value not in values:
                values.append(value)
            if len(values) >= limit:
                return values
    return values


def _heuristic_cases(doc: dict[str, Any], *, questions_per_chunk: int) -> list[dict[str, Any]]:
    title = str(doc.get("title") or "该制度")
    content = str(doc.get("content") or "")
    first_sentence = re.split(r"[。！？\n]", content, maxsplit=1)[0].strip() or content[:80]
    questions = [
        f"{title} 的核心要求是什么？",
        f"根据制度，{first_sentence[:40]} 这条规则是什么意思？",
        f"请解释 {title} 中和差旅相关的标准。",
    ]
    return [
        {
            "question": question,
            "expected_answer": first_sentence[:180],
            "keywords": _keywords_from_text(content),
        }
        for question in questions[:questions_per_chunk]
    ]


def _llm_cases(
    doc: dict[str, Any],
    *,
    questions_per_chunk: int,
    llm_base_url: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    content = str(doc.get("content") or "")
    prompt = [
        {
            "role": "system",
            "content": (
                "你是 RAG 评测集构造助手。请基于给定知识库 chunk 生成评测问题。"
                "只输出 JSON 数组，每个元素包含 question, expected_answer, keywords。"
                "要求：问题必须能被该 chunk 直接回答；不要引入 chunk 外的信息；"
                "keywords 选择 3-8 个用于答案校验的关键词。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"生成 {questions_per_chunk} 条。\n"
                f"chunk_id: {doc.get('id')}\n"
                f"title: {doc.get('title')}\n"
                f"doc_type: {doc.get('doc_type')}\n"
                f"content:\n{content[:1800]}"
            ),
        },
    ]
    raw = _call_llm(
        llm_base_url=llm_base_url,
        api_key=api_key,
        model=model,
        messages=prompt,
        timeout_s=timeout_s,
    )
    return _parse_json_array(raw)


def _safe_case_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "chunk")).strip("-")
    return text[:80] or "chunk"


def _normalize_generated_case(item: dict[str, Any], doc: dict[str, Any], index: int) -> dict[str, Any] | None:
    question = str(item.get("question") or "").strip()
    if not question:
        return None
    doc_id = str(doc.get("id") or "")
    keywords = item.get("keywords") if isinstance(item.get("keywords"), list) else []
    content = str(doc.get("content") or "")
    normalized_keywords = [str(value).strip() for value in keywords if str(value).strip()]
    if not normalized_keywords:
        normalized_keywords = _keywords_from_text(content)
    return {
        "id": f"auto-{_safe_case_id(doc_id)}-{index}",
        "question": question,
        "relevant_ids": [doc_id],
        "expected_doc_ids": [doc_id],
        "keywords": normalized_keywords[:8],
        "expected_answer": str(item.get("expected_answer") or "").strip(),
        "source_title": str(doc.get("title") or ""),
        "source_doc_type": str(doc.get("doc_type") or ""),
        "source_content_preview": content[:500],
        "review": {"status": "pending", "notes": ""},
    }


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    docs = _list_documents(args.base_url, limit=args.limit)
    if args.doc_type:
        docs = [doc for doc in docs if str(doc.get("doc_type") or "") == args.doc_type]
    docs = [
        doc
        for doc in docs
        if len(str(doc.get("content") or "").strip()) >= args.min_content_chars
    ][: args.max_chunks]

    cases: list[dict[str, Any]] = []
    for doc in docs:
        generated: list[dict[str, Any]] = []
        if args.generator == "llm":
            try:
                generated = _llm_cases(
                    doc,
                    questions_per_chunk=args.questions_per_chunk,
                    llm_base_url=args.llm_base_url,
                    api_key=args.llm_api_key,
                    model=args.llm_model,
                    timeout_s=args.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - eval builder should continue per chunk
                print(
                    f"LLM generation failed for {doc.get('id')}: {exc}; using heuristic fallback",
                    file=sys.stderr,
                )
        if not generated:
            generated = _heuristic_cases(doc, questions_per_chunk=args.questions_per_chunk)
        for index, item in enumerate(generated[: args.questions_per_chunk], start=1):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_generated_case(item, doc, index)
            if normalized:
                cases.append(normalized)
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "reviewer_decision",
        "corrected_question",
        "corrected_relevant_ids",
        "notes",
        "question",
        "relevant_ids",
        "keywords",
        "expected_answer",
        "source_title",
        "source_doc_type",
        "source_content_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "reviewer_decision": "",
                    "corrected_question": "",
                    "corrected_relevant_ids": "",
                    "notes": "",
                    "question": row["question"],
                    "relevant_ids": json.dumps(row["relevant_ids"], ensure_ascii=False),
                    "keywords": json.dumps(row["keywords"], ensure_ascii=False),
                    "expected_answer": row.get("expected_answer", ""),
                    "source_title": row.get("source_title", ""),
                    "source_doc_type": row.get("source_doc_type", ""),
                    "source_content_preview": row.get("source_content_preview", ""),
                }
            )


def main() -> int:
    _load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="evals/rag_cases_generated.jsonl")
    parser.add_argument("--review-csv", default="")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-chunks", type=int, default=80)
    parser.add_argument("--questions-per-chunk", type=int, default=2)
    parser.add_argument("--min-content-chars", type=int, default=30)
    parser.add_argument("--doc-type", default="")
    parser.add_argument("--generator", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.generator == "llm" and not args.llm_base_url:
        print("OPENAI_BASE_URL is required for --generator llm", file=sys.stderr)
        return 2

    cases = build_cases(args)
    output = Path(args.output)
    review_csv = Path(args.review_csv) if args.review_csv else output.with_suffix(".review.csv")
    write_jsonl(output, cases)
    write_review_csv(review_csv, cases)
    print(
        json.dumps(
            {
                "status": "ok",
                "case_count": len(cases),
                "output": str(output),
                "review_csv": str(review_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
