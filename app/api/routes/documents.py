from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.config import settings
from app.domain.schemas import (
    DocumentBatchDeleteRequest,
    DocumentIngestRequest,
    DocumentIngestResponse,
    LongDocumentIngestRequest,
)
from app.etl.pipeline import (
    chunk_document,
    enforce_document_chunk_token_limit,
    parse_plain_text,
    stable_chunk_id,
)
from app.etl.semantic_chunking import (
    expand_table_rows,
    group_pdf_articles,
    split_on_embedding_similarity,
)
from app.services.document_loader import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    DocumentLoadError,
    load_document_from_file,
)
from app.services.embeddings import EmbeddingService
from app.services.milvus_store import new_doc_id, utc_now

router = APIRouter(tags=["documents"])


def _require_embedding_configured() -> None:
    if not (settings.embedding_api_key or settings.openai_api_key):
        raise HTTPException(
            status_code=503,
            detail="请在 .env 中配置 EMBEDDING_API_KEY 或 OPENAI_API_KEY 后重启服务",
        )


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(body: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法写入知识库")
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if keyword_index is None or not keyword_index.connected:
        raise HTTPException(status_code=503, detail="关键词索引未连接，无法执行双路入库")

    _require_embedding_configured()
    embedder = EmbeddingService()
    vector = await embedder.embed_text(body.content)
    doc_id = new_doc_id()
    try:
        milvus.insert_vector(
            doc_id=doc_id,
            title=body.title,
            doc_type=body.doc_type,
            content=body.content,
            vector=vector,
        )
        await keyword_index.upsert_many(
            [
                {
                    "chunk_id": doc_id,
                    "parent_doc_id": doc_id,
                    "title": body.title,
                    "doc_type": body.doc_type,
                    "text": body.content,
                    "metadata": dict(body.metadata),
                }
            ]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"向量库写入失败: {exc}") from exc

    return DocumentIngestResponse(
        doc_id=doc_id,
        collection=getattr(milvus, "collection_name", "travel_knowledge"),
        inserted_at=utc_now(),
        vector_dim=len(vector),
        chunk_count=1,
        chunk_ids=[doc_id],
    )


@router.post("/documents/ingest-long", response_model=DocumentIngestResponse)
async def ingest_long_document(
    body: LongDocumentIngestRequest,
    request: Request,
) -> DocumentIngestResponse:
    return await _ingest_long_text(
        request=request,
        title=body.title,
        doc_type=body.doc_type,
        content=body.content,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        parent_doc_id=str(body.metadata.get("doc_id") or new_doc_id()),
        source_format=str(body.metadata.get("source_format") or ".txt"),
        source_metadata=dict(body.metadata),
    )


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    doc_type: Literal["policy", "sop", "city_guide", "other"] = Form("policy"),
    chunk_size: int = Form(3000, ge=200, le=12000),
    chunk_overlap: int = Form(450, ge=0, le=2000),
    doc_id: str | None = Form(None),
) -> DocumentIngestResponse:
    filename = file.filename or "document"
    try:
        data = await file.read()
        loaded = load_document_from_file(filename, data)
        content = loaded.text
    except DocumentLoadError as exc:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"{exc}。支持格式：{supported}") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"文件解析失败: {exc}") from exc
    if not content.strip():
        raise HTTPException(status_code=422, detail="未能从文件中提取到可入库文本")

    return await _ingest_long_text(
        request=request,
        title=title or filename,
        doc_type=doc_type,
        content=content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_doc_id=doc_id or new_doc_id(),
        source_format=Path(filename).suffix.lower(),
        source_metadata=loaded.metadata,
    )


async def _ingest_long_text(
    *,
    request: Request,
    title: str,
    doc_type: Literal["policy", "sop", "city_guide", "other"],
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    parent_doc_id: str,
    source_format: str,
    source_metadata: dict,
) -> DocumentIngestResponse:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法写入知识库")
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if keyword_index is None or not keyword_index.connected:
        raise HTTPException(status_code=503, detail="关键词索引未连接，无法执行双路入库")
    _ = chunk_overlap  # Compatibility-only input; non-PDF ingestion uses 15% overlap.

    _require_embedding_configured()
    text = parse_plain_text(content)
    embedder = EmbeddingService()
    if source_format == ".pdf" and settings.rag_semantic_pdf_enabled:
        chunks = chunk_document(
            text, source_format=source_format, max_chars=chunk_size,
            overlap_ratio=0, source_metadata=source_metadata,
        )
        chunks = expand_table_rows(group_pdf_articles(chunks))
        semantic_max = min(chunk_size, settings.rag_semantic_max_chars)
        semantic_target = min(semantic_max, settings.rag_semantic_target_chars)
        semantic_min = min(semantic_target, settings.rag_semantic_min_chars)
        chunks = await split_on_embedding_similarity(
            chunks, embedder, min_chars=semantic_min,
            target_chars=semantic_target, max_chars=semantic_max,
        )
    else:
        chunks = chunk_document(
            text, source_format=source_format, max_chars=chunk_size,
            overlap_ratio=0.15, source_metadata=source_metadata,
        )
    chunks = enforce_document_chunk_token_limit(
        chunks, max_tokens=settings.embedding_chunk_max_tokens
    )
    if not chunks:
        raise HTTPException(status_code=422, detail="文档内容为空，无法分块入库")

    contents = [chunk.content for chunk in chunks]
    chunk_ids = [
        stable_chunk_id(parent_doc_id, index, chunk.content)
        for index, chunk in enumerate(chunks)
    ]
    try:
        vectors = await embedder.embed_texts(contents)
        milvus.insert_vectors(
            doc_ids=chunk_ids,
            title=title,
            doc_type=doc_type,
            contents=contents,
            vectors=vectors,
        )
        await keyword_index.upsert_many(
            {
                "chunk_id": chunk_id,
                "parent_doc_id": parent_doc_id,
                "title": title,
                "doc_type": doc_type,
                "text": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk_id, chunk in zip(chunk_ids, chunks, strict=True)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"长文档分块入库失败: {exc}") from exc

    return DocumentIngestResponse(
        doc_id=parent_doc_id,
        collection=getattr(milvus, "collection_name", "travel_knowledge"),
        inserted_at=utc_now(),
        vector_dim=len(vectors[0]) if vectors else None,
        chunk_count=len(chunks),
        chunk_ids=chunk_ids,
    )


@router.get("/documents")
async def list_documents(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法列出知识库")
    try:
        docs = milvus.list_documents(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"列出知识库失败: {exc}") from exc
    return {"collection": getattr(milvus, "collection_name", "travel_knowledge"), "documents": docs}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> dict:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法删除知识")
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if keyword_index is None or not keyword_index.connected:
        raise HTTPException(status_code=503, detail="关键词索引未连接，无法同步删除知识")
    try:
        deleted = milvus.delete_document(doc_id=doc_id)
        await keyword_index.delete(doc_id=doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除知识失败: {exc}") from exc
    return {"deleted": deleted, "doc_id": doc_id}


@router.post("/documents/batch-delete")
async def batch_delete_documents(body: DocumentBatchDeleteRequest, request: Request) -> dict:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法删除知识")
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if keyword_index is None or not keyword_index.connected:
        raise HTTPException(status_code=503, detail="关键词索引未连接，无法同步删除知识")
    results: list[dict[str, int | str]] = []
    total_deleted = 0
    for doc_id in body.doc_ids:
        try:
            deleted = milvus.delete_document(doc_id=doc_id)
            await keyword_index.delete(doc_id=doc_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"删除知识失败: {doc_id}: {exc}") from exc
        total_deleted += deleted
        results.append({"doc_id": doc_id, "deleted": deleted})
    return {"deleted": total_deleted, "count": len(body.doc_ids), "results": results}


@router.delete("/documents")
async def delete_documents_by_title(
    request: Request,
    title: str = Query(..., min_length=1),
) -> dict:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法删除知识")
    keyword_index = getattr(request.app.state, "keyword_index", None)
    if keyword_index is None or not keyword_index.connected:
        raise HTTPException(status_code=503, detail="关键词索引未连接，无法同步删除知识")
    try:
        deleted = milvus.delete_document(title=title)
        await keyword_index.delete(title=title)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除知识失败: {exc}") from exc
    return {"deleted": deleted, "title": title}


@router.get("/documents/search")
async def search_documents(
    request: Request,
    q: str = Query(..., min_length=1, description="查询文本"),
    top_k: int = Query(5, ge=1, le=5),
) -> dict:
    retriever = getattr(request.app.state, "rag_retriever", None)
    if retriever is None or not retriever.connected:
        raise HTTPException(status_code=503, detail="混合检索依赖未连接，无法检索")

    _require_embedding_configured()
    embedder = EmbeddingService()
    vector = await embedder.embed_text(q)
    try:
        hits = await retriever.retrieve(q, vector, final_top_k=min(top_k, settings.rag_final_top_k))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc

    return {"query": q, "results": hits}
