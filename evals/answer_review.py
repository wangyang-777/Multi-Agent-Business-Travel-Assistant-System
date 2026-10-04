"""Create a human review sheet from chat evaluation and score explicit 0/1 labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "id", "case_type", "answerable", "question", "expected_answer", "answer_text",
    "relevant_ids", "retrieved_ids", "citation_ids", "answer_correct_0_or_1",
    "grounded_0_or_1", "citation_correct_0_or_1", "abstained_0_or_1", "notes",
]


def _rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("chat_sample"), list):
        return report["chat_sample"]
    payload = report.get("report", report)
    rows = payload.get("cases") if isinstance(payload, dict) else None
    return [row for row in rows if "answer_text" in row] if isinstance(rows, list) else []


def make_sheet(report_path: Path, output: Path) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _rows_from_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(row.get(key, []), ensure_ascii=False)
                    if key in {"relevant_ids", "retrieved_ids", "citation_ids"}
                    else row.get(key, "")
                )
                for key in FIELDS
            })
    return len(rows)


def _label(row: dict[str, str], key: str) -> int | None:
    raw = str(row.get(key) or "").strip()
    if not raw:
        return None
    if raw not in {"0", "1"}:
        raise ValueError(f"{row.get('id')}: {key} must be 0 or 1, got {raw!r}")
    return int(raw)


def score_sheet(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Review sheet has no cases")
    metrics = {
        "answer_accuracy": (True, "answer_correct_0_or_1"),
        "grounded_rate": (True, "grounded_0_or_1"),
        "citation_correct_rate": (True, "citation_correct_0_or_1"),
        "negative_abstention_rate": (False, "abstained_0_or_1"),
    }
    output: dict[str, Any] = {"case_count": len(rows)}
    for metric, (answerable, column) in metrics.items():
        subset = [
            row for row in rows
            if str(row.get("answerable") or "").strip().lower() == str(answerable).lower()
        ]
        labels = [_label(row, column) for row in subset]
        reviewed = [value for value in labels if value is not None]
        output[metric] = sum(reviewed) / len(reviewed) if reviewed else None
        output[f"{metric}_reviewed"] = len(reviewed)
        output[f"{metric}_total"] = len(subset)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--report", required=True)
    make.add_argument("--output", required=True)
    score = sub.add_parser("score")
    score.add_argument("--review-csv", required=True)
    score.add_argument("--output", default="")
    args = parser.parse_args()
    if args.command == "make":
        count = make_sheet(Path(args.report), Path(args.output))
        print(json.dumps({"review_rows": count, "output": args.output}, ensure_ascii=False))
    else:
        result = score_sheet(Path(args.review_csv))
        if args.output:
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
