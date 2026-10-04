from __future__ import annotations

import csv
import json

from evals.answer_review import make_sheet, score_sheet


def test_answer_review_scores_only_explicit_labels(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"report": {"cases": [
            {"id": "positive", "answerable": True, "question": "Q1", "answer_text": "A1"},
            {"id": "negative", "answerable": False, "question": "Q2", "answer_text": "unknown"},
        ]}}),
        encoding="utf-8",
    )
    review = tmp_path / "review.csv"
    assert make_sheet(report, review) == 2

    with review.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    rows[0]["answer_correct_0_or_1"] = "1"
    rows[0]["grounded_0_or_1"] = "0"
    rows[1]["abstained_0_or_1"] = "1"
    with review.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    score = score_sheet(review)
    assert score["answer_accuracy"] == 1.0
    assert score["grounded_rate"] == 0.0
    assert score["citation_correct_rate"] is None
    assert score["negative_abstention_rate"] == 1.0
