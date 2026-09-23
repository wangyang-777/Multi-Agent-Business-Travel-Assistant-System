from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.agent.orchestrator import TravelOrchestrator
from app.api.routes import chat as chat_routes
from app.api.routes import documents as documents_routes
from app.api.routes import health as health_routes
from app.api.routes import mcp as mcp_routes
from app.api.routes import sessions as sessions_routes
from app.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.rag.hybrid import HybridRAGRetriever
from app.core.rag.reranker import CrossEncoderReranker
from app.services.keyword_index import RedisKeywordIndex
from app.services.milvus_store import get_milvus_store

logger = get_logger(__name__)


def _setup_tracing() -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": settings.app_name}))
    trace.set_tracer_provider(provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    _setup_tracing()
    trace.get_tracer(__name__)

    app.state.redis = None
    try:
        app.state.redis = redis.from_url(settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        logger.info("redis.connected")
    except Exception as exc:
        logger.warning("redis.unavailable", error=str(exc))
        app.state.redis = None

    app.state.db_engine: Optional[AsyncEngine] = None
    try:
        app.state.db_engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        logger.info("database.engine_created")
    except Exception as exc:
        logger.warning("database.engine_failed", error=str(exc))

    store = get_milvus_store()
    store.connect()
    app.state.milvus = store
    if store.connected:
        logger.info("milvus.connected")
    else:
        logger.warning("milvus.unavailable")

    app.state.keyword_index = RedisKeywordIndex(app.state.redis)
    if store.connected and app.state.keyword_index.connected:
        try:
            existing_chunks = store.list_documents(limit=10000, content_limit=65530)
            await app.state.keyword_index.upsert_many(
                {
                    "chunk_id": str(item.get("id") or ""),
                    "parent_doc_id": _parent_document_id(str(item.get("id") or "")),
                    "title": item.get("title"),
                    "doc_type": item.get("doc_type"),
                    "text": item.get("content"),
                    "metadata": {"bootstrap_source": "milvus"},
                }
                for item in existing_chunks
            )
            logger.info("keyword_index.bootstrapped", chunks=len(existing_chunks))
        except Exception as exc:
            logger.warning("keyword_index.bootstrap_failed", error=str(exc))
    app.state.rag_retriever = HybridRAGRetriever(
        app.state.milvus,
        app.state.keyword_index,
        reranker=CrossEncoderReranker(model_name=settings.rag_reranker_model),
    )

    app.state.orchestrator = _create_orchestrator(
        redis_client=app.state.redis,
        document_store=app.state.milvus,
        rag_retriever=app.state.rag_retriever,
    )

    yield

    if app.state.redis is not None:
        await app.state.redis.close()
    eng = getattr(app.state, "db_engine", None)
    if eng is not None:
        await eng.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router, prefix="/api/v1")
    app.include_router(chat_routes.router, prefix="/api/v1")
    app.include_router(documents_routes.router, prefix="/api/v1")
    app.include_router(mcp_routes.router, prefix="/api/v1")
    app.include_router(sessions_routes.router, prefix="/api/v1")
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/app", StaticFiles(directory=str(static_dir), html=True), name="console")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": settings.app_name, "docs": "/docs", "app": "/app"}

    return app


def _create_orchestrator(
    redis_client: object | None = None,
    document_store: object | None = None,
    rag_retriever: object | None = None,
) -> TravelOrchestrator:
    if settings.agent_orchestrator_backend.lower() != "langgraph":
        return TravelOrchestrator(redis_client=redis_client)
    try:
        from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator

        return LangGraphTravelOrchestrator(
            redis_client=redis_client,
            document_store=document_store,
            rag_retriever=rag_retriever,
        )
    except ImportError as exc:
        logger.warning("langgraph.unavailable_fallback_legacy", error=str(exc))
        return TravelOrchestrator(redis_client=redis_client)


app = create_app()


def _parent_document_id(chunk_id: str) -> str:
    if chunk_id.startswith("chk_") and "_" in chunk_id[4:]:
        return chunk_id[4:].rsplit("_", 1)[0]
    return chunk_id
