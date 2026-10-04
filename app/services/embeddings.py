from __future__ import annotations

from app.config import settings
from app.utils.openai_client import create_openai_client
from app.utils.tokenization import truncate_text_by_tokens

_EMBED_BATCH_SIZE = 10


class EmbeddingService:
    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._client = create_openai_client(
            api_key=api_key or settings.embedding_api_key or settings.openai_api_key or "dummy",
            base_url=base_url or settings.embedding_base_url,
        )
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            kwargs = {
                "model": self._model,
                "input": [truncate_text_by_tokens(text, settings.embedding_max_tokens) for text in batch],
                "encoding_format": "float",
            }
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            resp = await self._client.embeddings.create(**kwargs)
            vectors.extend(list(item.embedding) for item in resp.data)
        return vectors
