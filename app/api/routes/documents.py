from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.config import settings
from app.domain.schemas import (
    DocumentBatchDeleteRequest,
    DocumentIngestRequest,
    DocumentIngestResponse,
    LongDocumentIngestRequest,
)
from app.etl.pipeline import enforce_chunk_token_limit, chunk_text, parse_plain_text, stable_chunk_id
from app.services.document_loader import SUPPORTED_DOCUMENT_EXTENSIONS, DocumentLoadError, load_document_from_file
from app.services.embeddings import EmbeddingService
from app.services.milvus_store import new_doc_id, utc_now

router = APIRouter(tags=["documents"])


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(body: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法写入知识库")

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
    )


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    doc_type: Literal["policy", "sop", "city_guide", "other"] = Form("policy"),
    chunk_size: int = Form(800, ge=200, le=3000),
    chunk_overlap: int = Form(120, ge=0, le=1000),
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
) -> DocumentIngestResponse:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法写入知识库")
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap 必须小于 chunk_size")

    text = parse_plain_text(content)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    chunks = enforce_chunk_token_limit(chunks, max_tokens=settings.embedding_chunk_max_tokens)
    if not chunks:
        raise HTTPException(status_code=422, detail="文档内容为空，无法分块入库")

    chunk_ids = [stable_chunk_id(parent_doc_id, index, chunk) for index, chunk in enumerate(chunks)]
    embedder = EmbeddingService()
    try:
        vectors = await embedder.embed_texts(chunks)
        milvus.insert_vectors(
            doc_ids=chunk_ids,
            title=title,
            doc_type=doc_type,
            contents=chunks,
            vectors=vectors,
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
    try:
        deleted = milvus.delete_document(doc_id=doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除知识失败: {exc}") from exc
    return {"deleted": deleted, "doc_id": doc_id}


@router.post("/documents/batch-delete")
async def batch_delete_documents(body: DocumentBatchDeleteRequest, request: Request) -> dict:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法删除知识")
    results: list[dict[str, int | str]] = []
    total_deleted = 0
    for doc_id in body.doc_ids:
        try:
            deleted = milvus.delete_document(doc_id=doc_id)
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
    try:
        deleted = milvus.delete_document(title=title)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除知识失败: {exc}") from exc
    return {"deleted": deleted, "title": title}


@router.get("/documents/search")
async def search_documents(
    request: Request,
    q: str = Query(..., min_length=1, description="查询文本"),
    top_k: int = Query(5, ge=1, le=20),
) -> dict:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法检索")

    embedder = EmbeddingService()
    vector = await embedder.embed_text(q)
    try:
        hits = milvus.search(vector, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc

    return {"query": q, "results": hits}
