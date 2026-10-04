from __future__ import annotations

import json

from evals.apply_rag_review import apply_review
from evals.build_rag_cases import _normalize_generated_case, _parse_json_array
from evals.run_rag_eval import (
    _evidence_group_metrics,
    _matches_expected,
    _parse_top_ks,
    _retrieval_metrics,
    load_cases,
)


def test_load_cases_supports_relevant_ids(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "question": "staff 去上海酒店标准是多少？",
                "relevant_ids": ["doc-a", "doc-b"],
                "keywords": ["800", "酒店"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].case_id == "case-1"
    assert cases[0].relevant_ids == ["doc-a", "doc-b"]
    assert cases[0].expected_doc_ids == ["doc-a", "doc-b"]


def test_retrieval_metrics_calculates_precision_recall_hit() -> None:
    results = [
        {"id": "doc-a"},
        {"id": "noise-1"},
        {"id": "doc-b"},
        {"id": "noise-2"},
    ]

    metrics = _retrieval_metrics(results, ["doc-a", "doc-b", "doc-c"], top_k=3)

    assert metrics["hit@3"] is True
    assert metrics["precision@3"] == 2 / 3
    assert metrics["recall@3"] == 2 / 3


def test_retrieval_matches_exact_chunk_id_without_matching_content() -> None:
    assert _matches_expected({"chunk_id": "chunk-a", "content": "chunk-b"}, ["chunk-a"])
    assert not _matches_expected({"chunk_id": "chunk-a", "content": "chunk-b"}, ["chunk-b"])
    assert not _matches_expected({"chunk_id": "chunk-a-extra"}, ["chunk-a"])


def test_evidence_groups_accept_alternatives_and_require_multisource() -> None:
    groups = [["table", "duplicate-text"], ["seasonal-appendix"]]
    top_one = _evidence_group_metrics([{"chunk_id": "duplicate-text"}], groups, top_k=1)
    top_two = _evidence_group_metrics(
        [{"chunk_id": "duplicate-text"}, {"chunk_id": "seasonal-appendix"}],
        groups,
        top_k=2,
    )

    assert top_one == {"evidence_coverage@1": 0.5, "all_evidence@1": False}
    assert top_two == {"evidence_coverage@2": 1.0, "all_evidence@2": True}


def test_parse_top_ks_sorts_and_deduplicates() -> None:
    assert _parse_top_ks("5,1,3,3", 5) == [1, 3, 5]


def test_build_cases_parses_json_array_from_markdown() -> None:
    parsed = _parse_json_array(
        """
        ```json
        [{"question":"Q","expected_answer":"A","keywords":["K"]}]
        ```
        """
    )

    assert parsed == [{"question": "Q", "expected_answer": "A", "keywords": ["K"]}]


def test_normalize_generated_case_binds_chunk_id() -> None:
    doc = {
        "id": "chunk-1",
        "title": "policy",
        "doc_type": "policy",
        "content": "staff 酒店标准 800 CNY",
    }

    case = _normalize_generated_case(
        {"question": "staff 酒店标准是多少？", "expected_answer": "800 CNY"},
        doc,
        1,
    )

    assert case is not None
    assert case["relevant_ids"] == ["chunk-1"]
    assert case["expected_doc_ids"] == ["chunk-1"]
    assert "800 CNY" in case["source_content_preview"]


def test_apply_rag_review_keeps_fixes_and_drops(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "keep", "question": "Q1", "relevant_ids": ["a"]}),
                json.dumps({"id": "fix", "question": "Q2", "relevant_ids": ["b"]}),
                json.dumps({"id": "drop", "question": "Q3", "relevant_ids": ["c"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    review_path = tmp_path / "review.csv"
    review_path.write_text(
        "\n".join(
            [
                "id,reviewer_decision,corrected_question,corrected_relevant_ids,notes",
                "keep,keep,,,ok",
                "fix,fix,Q2 corrected,\"[\"\"b\"\",\"\"d\"\"]\",needs another chunk",
                "drop,drop,,,too broad",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    curated = apply_review(cases_path=cases_path, review_csv_path=review_path)

    assert [item["id"] for item in curated] == ["keep", "fix"]
    assert curated[0]["review"]["status"] == "approved"
    assert curated[1]["question"] == "Q2 corrected"
    assert curated[1]["relevant_ids"] == ["b", "d"]


def test_apply_rag_review_does_not_approve_blank_decision(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text('{"id":"unreviewed","question":"Q","relevant_ids":["a"]}\n', encoding="utf-8")
    review_path = tmp_path / "review.csv"
    review_path.write_text("id,reviewer_decision\nunreviewed,\n", encoding="utf-8")

    assert apply_review(cases_path=cases_path, review_csv_path=review_path) == []


def test_eval_preserves_stage_data_and_configurable_chat_timeout(monkeypatch):
    from evals import run_rag_eval

    timeouts = []
    evidence = {"facts": {"p_rate": {"value": "135", "chunk_ids": ["policy"]}}}
    stages = [{"stage": "final", "answer": "405元[1]"}]

    def request(method, url, *, payload=None, timeout_s=30.0):
        if method == "GET":
            return {"results": [{"chunk_id": "policy"}]}
        timeouts.append(timeout_s)
        return {"choices": [{"message": {"content": "405元[1]"}}],
                "citations": [{"chunk_id": "policy"}], "rag_evidence": evidence,
                "rag_stages": stages, "verification": {"passed": True, "question_answered": True},
                "usage": {"total_tokens": 99}}

    monkeypatch.setattr(run_rag_eval, "_request_json", request)
    report = run_rag_eval.evaluate(
        "http://localhost", [run_rag_eval.RagCase("a", "3天补助？", ["policy"], ["405"])],
        top_ks=[1], include_chat=True, chat_timeout_s=420,
    )
    assert timeouts == [420]
    assert report["cases"][0]["rag_evidence"] == evidence
    assert report["cases"][0]["rag_stages"] == stages
    assert report["cases"][0]["usage"]["total_tokens"] == 99
