from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "travel_knowledge"


def _load_pymilvus() -> tuple[Any, ...]:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    return (Collection, CollectionSchema, DataType, FieldSchema, connections, utility)


@dataclass
class MilvusDocumentStore:
    host: str
    port: int
    collection_name: str = COLLECTION_NAME
    _collection: Any = None
    _connected: bool = field(default=False, init=False)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if self._connected and self._collection is not None:
            return True
        try:
            (
                MilvusCollection,
                MilvusCollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            ) = _load_pymilvus()
            alias = "default"
            connections.connect(alias=alias, host=self.host, port=str(self.port))
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(
                        name="embedding", dtype=DataType.FLOAT_VECTOR,
                        dim=settings.embedding_dimensions,
                    ),
                ]
                schema = MilvusCollectionSchema(fields, description="Business travel knowledge base")
                col = MilvusCollection(name=self.collection_name, schema=schema)
                index = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                }
                col.create_index(field_name="embedding", index_params=index)
            self._collection = MilvusCollection(self.collection_name)
            self._collection.load()
            self._connected = True
            return True
        except Exception as exc:
            logger.warning("milvus.connect_failed", error=str(exc))
            self._connected = False
            self._collection = None
            return False

    def insert_vector(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        content: str,
        vector: List[float],
    ) -> None:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        self._collection.insert(
            [
                [doc_id],
                [title[:512]],
                [doc_type[:32]],
                [content[:65530]],
                [vector],
            ]
        )
        self._collection.flush()

    def insert_vectors(
        self,
        *,
        doc_ids: List[str],
        title: str,
        doc_type: str,
        contents: List[str],
        vectors: List[List[float]],
    ) -> None:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        if not (len(doc_ids) == len(contents) == len(vectors)):
            raise ValueError("doc_ids, contents and vectors length mismatch")
        if not doc_ids:
            return
        self._collection.insert(
            [
                doc_ids,
                [title[:512]] * len(doc_ids),
                [doc_type[:32]] * len(doc_ids),
                [content[:65530] for content in contents],
                vectors,
            ]
        )
        self._collection.flush()

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        self._collection.load()
        res = self._collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["title", "doc_type", "content"],
        )
        hits: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent: Dict[str, Any] = {}
            raw = getattr(hit, "entity", None)
            if raw is not None:
                if hasattr(raw, "to_dict"):
                    ent = raw.to_dict()
                elif isinstance(raw, dict):
                    ent = raw
                else:
                    try:
                        ent = dict(raw)
                    except Exception:
                        ent = {}
            if "entity" in ent and isinstance(ent["entity"], dict):
                ent = ent["entity"]
            dist = getattr(hit, "distance", None)
            hits.append(
                {
                    "id": getattr(hit, "id", None),
                    "score": float(dist) if dist is not None else 0.0,
                    "title": ent.get("title"),
                    "doc_type": ent.get("doc_type"),
                    "content": str(ent.get("content") or "")[:2000],
                }
            )
        return hits

    def list_documents(
        self, limit: int = 100, *, content_limit: int = 2000
    ) -> List[Dict[str, Any]]:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        self._collection.load()
        rows = self._collection.query(
            expr="id != ''",
            output_fields=["id", "title", "doc_type", "content"],
            limit=limit,
        )
        return [
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "doc_type": row.get("doc_type"),
                "content": str(row.get("content") or "")[:content_limit],
            }
            for row in rows
        ]

    def delete_document(self, *, doc_id: str | None = None, title: str | None = None) -> int:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        if not doc_id and not title:
            raise ValueError("doc_id or title is required")
        if doc_id:
            safe_id = str(doc_id).replace("\\", "\\\\").replace('"', '\\"')
            expr = f'id == "{safe_id}" or id like "chk_{safe_id}_%"'
        else:
            safe_title = str(title).replace("\\", "\\\\").replace('"', '\\"')
            expr = f'title == "{safe_title}"'
        result = self._collection.delete(expr)
        self._collection.flush()
        delete_count = getattr(result, "delete_count", None)
        return int(delete_count) if delete_count is not None else 0


def get_milvus_store() -> MilvusDocumentStore:
    return MilvusDocumentStore(host=settings.milvus_host, port=settings.milvus_port)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_doc_id() -> str:
    return str(uuid.uuid4())
