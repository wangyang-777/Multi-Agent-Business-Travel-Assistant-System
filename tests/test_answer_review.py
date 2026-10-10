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


def test_pending_referrals_are_separate_from_answer_errors(tmp_path):
    from evals.answer_review import FIELDS
    p = tmp_path / "review.csv"
    with p.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader()
        writer.writerows([
            {"id": "auto", "answerable": True, "human_review_required": False, "answer_correct_0_or_1": 1},
            {"id": "pending", "answerable": True, "human_review_required": True, "review_submitted": True, "answer_correct_0_or_1": 0, "review_appropriate_0_or_1": 1},
            {"id": "reviewed", "answerable": True, "human_review_required": True, "review_submitted": True, "reviewed_outcome_correct_0_or_1": 1, "review_duration_seconds": 30},
        ])
    score = score_sheet(p)
    assert score["answer_accuracy"] == 1
    assert score["answer_accuracy_total"] == 1
    assert score["manual_review_rate"] == 2 / 3
    assert score["review_appropriateness_rate"] == 1
    assert score["post_review_outcome_accuracy"] == 1
    assert score["post_review_outcome_accuracy_reviewed"] == 1
    assert score["review_duration_mean_seconds"] == 30
