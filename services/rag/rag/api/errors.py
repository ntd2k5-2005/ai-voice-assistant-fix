"""HTTP mapping for the stable RAG exception taxonomy."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

LOGGER = logging.getLogger("rag.api")


from ..errors import (
    ChunkingError,
    ConcurrentUpdateError,
    ConfigurationError,
    EmbeddingIdentityError,
    EmptyDocumentError,
    EncryptedDocumentError,
    LifecycleError,
    ModelError,
    OCRError,
    ParserError,
    RagError,
    RetrievalError,
    SourceTooLargeError,
    StorageConsistencyError,
    StorageError,
    UnsupportedSourceError,
)


def _response(
    status_code: int,
    exc: Exception | None = None,
    *,
    code: str | None = None,
    message: str | None = None,
    details=None,
) -> JSONResponse:
    payload = {
        "code": code or (type(exc).__name__ if exc is not None else "api_error"),
        "message": message if message is not None else str(exc or "request failed"),
    }
    if details is not None:
        payload["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, exc: RequestValidationError):
        return _response(
            422,
            code="request_validation_error",
            message="Request validation failed",
            details=jsonable_encoder(exc.errors(), custom_encoder={ValueError: str, TypeError: str}),
        )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code") or "http_error")
            message = str(detail.get("message") or detail)
            details = detail.get("details")
        else:
            code = {400: "bad_request", 404: "not_found", 409: "conflict", 413: "payload_too_large", 415: "unsupported_media_type", 422: "unprocessable_content", 503: "service_unavailable"}.get(int(exc.status_code), "http_error")
            message = str(detail)
            details = None
        return _response(
            int(exc.status_code),
            code=code,
            message=message,
            details=details,
        )

    @app.exception_handler(SourceTooLargeError)
    async def source_too_large(_: Request, exc: SourceTooLargeError):
        return _response(413, exc)

    @app.exception_handler(UnsupportedSourceError)
    async def unsupported_source(_: Request, exc: UnsupportedSourceError):
        return _response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, exc)

    for exception_type in (EncryptedDocumentError, EmptyDocumentError, OCRError, ParserError, ChunkingError):
        app.add_exception_handler(
            exception_type,
            lambda request, exc, code=422: _response(code, exc),
        )

    for exception_type in (ConcurrentUpdateError, EmbeddingIdentityError, StorageConsistencyError):
        app.add_exception_handler(
            exception_type,
            lambda request, exc, code=status.HTTP_409_CONFLICT: _response(code, exc),
        )

    @app.exception_handler(ConfigurationError)
    async def configuration_error(_: Request, exc: ConfigurationError):
        return _response(status.HTTP_400_BAD_REQUEST, exc)

    for exception_type in (ModelError, StorageError, LifecycleError, RetrievalError):
        app.add_exception_handler(
            exception_type,
            lambda request, exc, code=status.HTTP_503_SERVICE_UNAVAILABLE: _response(code, exc),
        )

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return _response(status.HTTP_400_BAD_REQUEST, exc)

    @app.exception_handler(TypeError)
    async def type_error(_: Request, exc: TypeError):
        return _response(status.HTTP_400_BAD_REQUEST, exc)

    @app.exception_handler(RagError)
    async def rag_error(_: Request, exc: RagError):
        return _response(status.HTTP_500_INTERNAL_SERVER_ERROR, exc)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception):
        LOGGER.exception("Unhandled RAG API error", exc_info=exc)
        return _response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_server_error",
            message="Internal server error",
        )


__all__ = ["install_exception_handlers"]
