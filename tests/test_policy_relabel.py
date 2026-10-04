from __future__ import annotations

import pytest

from evals.relabel_policy_cases import relabel_cases


def test_relabel_accepts_table_when_duplicate_body_was_removed() -> None:
    case = {
        "id": "hotel", "answerable": True,
        "evidence": [
            {"group": 1, "page": 3, "block": "table", "quote": "北京 | 600 元"},
            {"group": 1, "page": 3, "block": "paragraph", "quote": "北京 600 元"},
        ],
    }
    documents = [{
        "chunk_id": "table-new", "text": "| 北京 | 600 元 |",
        "metadata": {"page_numbers": [3], "block_types": ["table"]},
    }]

    result = relabel_cases([case], documents)[0]

    assert result["relevant_ids"] == ["table-new"]
    assert result["relevance_groups"] == [["table-new"]]


def test_relabel_requires_each_evidence_group() -> None:
    case = {
        "id": "multihop", "answerable": True,
        "evidence": [
            {"group": 1, "page": 3, "block": "table", "quote": "北京"},
            {"group": 2, "page": 7, "block": "table", "quote": "旺季"},
        ],
    }
    documents = [{
        "chunk_id": "table-new", "text": "北京",
        "metadata": {"page_numbers": [3], "block_types": ["table"]},
    }]

    with pytest.raises(ValueError, match="group"):
        relabel_cases([case], documents)


def test_relabel_accepts_verbatim_rule_repeated_as_table_context() -> None:
    case = {
        "id": "hotel-context", "answerable": True,
        "evidence": [{
            "group": 1, "page": 3, "block": "paragraph",
            "quote": "原则上不得入住五星级酒店",
        }],
    }
    documents = [{
        "chunk_id": "table-row", "text": "[Context] 原则上不得入住 五星级酒店",
        "metadata": {"page_numbers": [3], "block_types": ["table"]},
    }]

    assert relabel_cases([case], documents)[0]["relevant_ids"] == ["table-row"]
