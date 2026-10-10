from __future__ import annotations

from app.etl.pipeline import chunk_document


def test_markdown_preserves_heading_path_and_code_block() -> None:
    text = """# 差旅制度

## 酒店标准

员工住宿应遵守城市差标。

```python
def calculate_limit(city: str) -> int:
    return 800
```
"""

    chunks = chunk_document(text, source_format=".md", max_chars=80, overlap_ratio=0.15)

    hotel_chunks = [chunk for chunk in chunks if "酒店标准" in chunk.metadata["heading_path"]]
    assert hotel_chunks
    assert all(chunk.metadata["heading_path"] == ["差旅制度", "酒店标准"] for chunk in hotel_chunks)
    code_chunks = [chunk for chunk in chunks if "code" in chunk.metadata["block_types"]]
    assert len(code_chunks) == 1
    assert "def calculate_limit" in code_chunks[0].content
    assert "return 800" in code_chunks[0].content


def test_faq_pairs_are_indivisible_and_do_not_overlap() -> None:
    text = """问题：酒店超标怎么办？
答案：提交超标说明并走人工审批。

问题：发票有什么要求？
答案：必须提供真实合规的原始票据。
"""

    chunks = chunk_document(text, source_format=".txt", max_chars=40, overlap_ratio=0.15)

    assert len(chunks) == 2
    assert all(chunk.metadata["block_types"] == ["faq"] for chunk in chunks)
    assert all("问题：" in chunk.content and "答案：" in chunk.content for chunk in chunks)
    assert all("overlap_chars" not in chunk.metadata for chunk in chunks)


def test_pdf_groups_short_paragraphs_without_crossing_page_or_table() -> None:
    text = """[Page 1]
第一条 出差须事前审批。

第二条 住宿按标准报销。

[Page 2]
[Table 2-1]
| 人员 | 飞机 |
| --- | --- |
| L1 | 公务舱 |

第三条 其他人员乘经济舱。
"""
    chunks = chunk_document(text, source_format=".pdf", max_chars=300, overlap_ratio=0)

    assert len(chunks) == 3
    assert "第一条" in chunks[0].content and "第二条" in chunks[0].content
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].metadata["block_types"] == ["table"]
    assert chunks[2].metadata["page_number"] == 2
    assert "第三条" in chunks[2].content
