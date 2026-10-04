from __future__ import annotations

from app.etl.pipeline import DocumentChunk
from app.etl.semantic_chunking import (
    expand_table_rows,
    group_pdf_articles,
    split_on_embedding_similarity,
)
from app.services.document_parser import _body_text_without_tables, _needs_ocr, _table_context


def test_article_grouping_joins_page_continuation_and_keeps_table_separate() -> None:
    chunks = [
        DocumentChunk("第一条 审批规则。\n\n第二条 报销凭据要求。", {"page_number": 1, "block_types": ["paragraph"]}),
        DocumentChunk("[Table 2-1]\n| 人员 | 舱位 |", {"page_number": 2, "block_types": ["table"]}),
        DocumentChunk("凭据必须真实。\n\n第三条 住宿标准。", {"page_number": 2, "block_types": ["paragraph"]}),
    ]

    grouped = group_pdf_articles(chunks)
    second = next(chunk for chunk in grouped if chunk.metadata.get("article_no") == "第二条")

    assert "报销凭据要求" in second.content and "凭据必须真实" in second.content
    assert second.metadata["page_numbers"] == [1, 2]
    assert any("table" in chunk.metadata["block_types"] for chunk in grouped)
    assert all("第三条" not in chunk.content for chunk in grouped if chunk.metadata.get("article_no") == "第二条")


def test_pdf_without_articles_keeps_page_boundaries() -> None:
    chunks = [
        DocumentChunk("第一页内容。", {"page_number": 1, "block_types": ["paragraph"]}),
        DocumentChunk("第二页内容。", {"page_number": 2, "block_types": ["paragraph"]}),
    ]

    grouped = group_pdf_articles(chunks)

    assert [chunk.metadata["page_numbers"] for chunk in grouped] == [[1], [2]]


async def test_embedding_similarity_selects_low_similarity_sentence_boundary() -> None:
    class FakeEmbedder:
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 1.0] if "酒店" in text or "住宿" in text else [1.0, 0.0] for text in texts]

    text = "第一条 " + "审批" * 40 + "。" + "报销" * 40 + "。" + "酒店" * 40 + "。" + "住宿" * 40 + "。"
    chunk = DocumentChunk(text, {"article_no": "第一条", "block_types": ["paragraph"]})

    parts = await split_on_embedding_similarity(
        [chunk], FakeEmbedder(), min_chars=100, target_chars=180, max_chars=300
    )

    assert len(parts) == 2
    assert "报销" in parts[0].content and "酒店" not in parts[0].content
    assert "酒店" in parts[1].content and "住宿" in parts[1].content
    assert parts[0].metadata["semantic_boundary_similarity"] == 0.0


def test_pdf_table_text_is_not_duplicated_in_body() -> None:
    class FakePage:
        def get_text(self, mode: str, *, sort: bool) -> list[tuple]:
            assert mode == "blocks" and sort
            return [
                (0, 0, 100, 20, "第七条 交通规则。", 0, 0),
                (0, 30, 100, 50, "L1 公务舱", 1, 0),
            ]

    body = _body_text_without_tables(FakePage(), [(0, 25, 100, 55)])

    assert "第七条" in body
    assert "L1 公务舱" not in body


def test_table_rows_retain_header_and_merged_group_context() -> None:
    table = DocumentChunk(
        "[Table 7-1]\n| 省份 | 旺季地区 | 旺季期间 |\n| --- | --- | --- |\n"
        "| 海南 | 海口市 | 11-2 月 |\n|  | 三亚市 | 10-4 月 |",
        {"page_number": 7, "block_types": ["table"]},
    )

    chunks = expand_table_rows([table])

    assert len(chunks) == 3
    assert chunks[0] is table
    assert "旺季地区" in chunks[2].content
    assert "上级分组：海南" in chunks[2].content
    assert "三亚市 | 10-4 月" in chunks[2].content
    assert chunks[2].metadata["table_row_index"] == 2


def test_text_only_table_page_does_not_require_ocr() -> None:
    assert not _needs_ocr("", ["| 城市 | 旺季期间 |\n" * 5], 0)
    assert _needs_ocr("", [], 0)


def test_table_context_starts_at_nearest_article_heading() -> None:
    class FakePage:
        def get_text(self, mode: str, *, sort: bool) -> list[tuple]:
            assert mode == "blocks" and sort
            return [
                (0, 1, 100, 20, "第十一条 旧规则。", 0, 0),
                (0, 25, 100, 45, "第十二条 非协议酒店住宿限额如下：", 1, 0),
                (0, 50, 100, 65, "人员选择标准间。", 2, 0),
                (0, 80, 100, 100, "表格中的内容", 3, 0),
            ]

    context = _table_context(FakePage(), (0, 75, 100, 110))

    assert context.startswith("第十二条")
    assert "人员选择标准间" in context
    assert "旧规则" not in context and "表格中的内容" not in context
