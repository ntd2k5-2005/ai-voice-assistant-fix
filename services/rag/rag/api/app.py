"""FastAPI transport adapter for :class:`rag.service.RagModule`."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .errors import install_exception_handlers
from .routes import health_router, knowledge_router, retrieval_router
from .settings import ApiSettings


ModuleFactory = Callable[[], Any]


def _default_module_factory():
    # Heavy model/storage imports remain behind service startup, not API import.
    from ..service import RagModule
    from ..environment import config_from_environment

    return RagModule(config_from_environment())


def create_app(
    *,
    module_factory: ModuleFactory | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    factory = module_factory or _default_module_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        module = factory()
        app.state.rag_module = module
        try:
            yield
        finally:
            app.state.rag_module = None
            close = getattr(module, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="Robust RAG Service",
        version="1.0",
        description=(
            "HTTP adapter for knowledge ingestion/management and retrieval. "
            "LLM answer generation is intentionally outside this service."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.api_settings = settings
    install_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(knowledge_router)
    app.include_router(retrieval_router)
    return app


app = create_app()


__all__ = ["app", "create_app"]
