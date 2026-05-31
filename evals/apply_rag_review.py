from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows[str(item["id"])] = item
    return rows


def _parse_ids(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def apply_review(*, cases_path: Path, review_csv_path: Path) -> list[dict[str, Any]]:
    cases = _read_jsonl(cases_path)
    curated: list[dict[str, Any]] = []
    with review_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("id") or "")
            if case_id not in cases:
                continue
            decision = str(row.get("reviewer_decision") or "keep").strip().lower()
            if decision in {"drop", "reject", "删除", "丢弃"}:
                continue

            item = dict(cases[case_id])
            corrected_question = str(row.get("corrected_question") or "").strip()
            corrected_ids = _parse_ids(str(row.get("corrected_relevant_ids") or ""))
            if decision in {"fix", "修正"} or corrected_question or corrected_ids:
                if corrected_question:
                    item["question"] = corrected_question
                if corrected_ids:
                    item["relevant_ids"] = corrected_ids
                    item["expected_doc_ids"] = corrected_ids
                item["review"] = {
                    "status": "fixed",
                    "notes": str(row.get("notes") or "").strip(),
                }
            else:
                item["review"] = {
                    "status": "approved",
                    "notes": str(row.get("notes") or "").strip(),
                }
            curated.append(item)
    return curated


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/rag_cases_generated.jsonl")
    parser.add_argument("--review-csv", default="evals/rag_cases_generated.review.csv")
    parser.add_argument("--output", default="evals/rag_cases_curated.jsonl")
    args = parser.parse_args()

    rows = apply_review(cases_path=Path(args.cases), review_csv_path=Path(args.review_csv))
    write_jsonl(Path(args.output), rows)
    print(
        json.dumps(
            {"status": "ok", "case_count": len(rows), "output": args.output},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
