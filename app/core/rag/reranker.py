"""Cross-encoder reranking for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@dataclass(slots=True)
class RerankedChunk:
    chunk: RetrievedChunk
    cross_encoder_score: float


class CrossEncoderReranker:
    """Reranks chunks with a sentence-transformers cross-encoder (query-document pairs)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder as CE

            self._model = CE(self._model_name, device=self._device)
        return self._model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        model = self._load()
        pairs = [[query, c.text] for c in chunks]
        scores: list[float] = []
        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            raw = model.predict(batch, convert_to_numpy=True)
            scores.extend(float(x) for x in raw.tolist())
        out = [RerankedChunk(chunk=c, cross_encoder_score=s) for c, s in zip(chunks, scores, strict=True)]
        out.sort(key=lambda r: r.cross_encoder_score, reverse=True)
        return out if top_k is None else out[:top_k]

    def rerank_as_chunks(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        ranked = self.rerank(query, chunks, top_k=top_k)
        merged: list[RetrievedChunk] = []
        for r in ranked:
            meta = dict(r.chunk.metadata)
            meta["cross_encoder_score"] = r.cross_encoder_score
            merged.append(
                RetrievedChunk(
                    chunk_id=r.chunk.chunk_id,
                    text=r.chunk.text,
                    score=r.cross_encoder_score,
                    source=r.chunk.source,
                    metadata=meta,
                )
            )
        return merged
