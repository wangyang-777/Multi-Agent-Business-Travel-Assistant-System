from __future__ import annotations

from app.etl.pipeline import chunk_text, enforce_chunk_token_limit, parse_plain_text
from app.utils.tokenization import count_text_tokens


def test_chunk_text_keeps_policy_sentences_intact() -> None:
    text = parse_plain_text(
        """
        第一章 总则
        员工应至少提前7天预订差旅行程。低于提前7天预订可能产生附加费，并需要说明原因。

        第二章 酒店标准
        staff 员工去上海出差，酒店标准为每晚不超过800 CNY。超出标准需要提交审批。
        """
    )

    chunks = chunk_text(text, chunk_size=90, overlap=0)

    joined = "\n".join(chunks)
    assert "员工应至少提前7天预订差旅行程。" in joined
    assert "staff 员工去上海出差，酒店标准为每晚不超过800 CNY。" in joined
    assert all(not chunk.endswith("员工应至少提前7天预订差旅") for chunk in chunks)


def test_chunk_text_splits_oversized_unit_on_punctuation() -> None:
    text = "规则：" + "员工需要遵守差旅制度，" * 30 + "最终以公司制度为准。"

    chunks = chunk_text(text, chunk_size=120, overlap=0)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert any(chunk.endswith("，") or chunk.endswith("。") for chunk in chunks[:-1])


def test_chunk_text_adds_overlap_without_breaking_current_sentence_start() -> None:
    text = "第一条 员工应提前预订。\n\n第二条 酒店标准按城市等级执行。\n\n第三条 超额需要审批。"

    chunks = chunk_text(text, chunk_size=28, overlap=20)

    assert len(chunks) >= 2
    assert "第二条 酒店标准按城市等级执行。" in "\n".join(chunks)


def test_overlap_uses_complete_sentence_not_character_tail() -> None:
    text = (
        "第一条 出差费用报销遵循谁受益谁承担的原则。"
        "各部门应通过严格执行出差事前审批制度，确定出差费用的受益部门。"
        "若出差受益部门与员工所在部门不一致，出差申请经受益部门权签人审批同意。"
        "第二条 票据要求。所有住宿、交通费用报销时须提供真实合规的原始票据。"
    )

    chunks = chunk_text(text, chunk_size=95, overlap=35)

    assert len(chunks) >= 2
    assert not chunks[1].startswith("度，")
    assert not chunks[1].startswith("况，")
    assert chunks[1].startswith("第二条") or chunks[1].startswith("若出差受益部门")


def test_enforce_chunk_token_limit_splits_oversized_chunks() -> None:
    chunks = ["第一条 " + "差旅标准需要遵守公司制度。" * 80]

    limited = enforce_chunk_token_limit(chunks, max_tokens=80)

    assert len(limited) > 1
    assert all(count_text_tokens(chunk) <= 80 for chunk in limited)
