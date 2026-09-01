from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from ...errors import SourceTooLargeError

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status

from ..dependencies import get_rag_module
from ..schemas import (
    DeleteResponse,
    IngestionResponse,
    KnowledgeDetailResponse,
    KnowledgeListResponse,
    RecoveryResponse,
    TextIngestionRequest,
)
from ..serialization import (
    delete_to_response,
    ingestion_to_response,
    knowledge_detail_to_response,
    knowledge_list_to_response,
)


router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])


def _metadata(metadata_json: str | None) -> dict[str, Any]:
    if not metadata_json:
        return {}
    try:
        value = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="metadata_json must contain a valid JSON object",
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=422,
            detail="metadata_json must contain a JSON object",
        )
    return value


def _default_document_id(file: UploadFile) -> str:
    filename = (file.filename or "").strip()
    stem = Path(filename).stem.strip() if filename else ""
    if not stem:
        raise HTTPException(
            status_code=422,
            detail="document_id is required when the uploaded file has no usable filename",
        )
    return stem


def _ingest_file(
    *,
    module,
    file: UploadFile,
    document_id: str | None,
    metadata_json: str | None,
    operation_id: str | None,
    debug: bool,
    force: bool,
) -> IngestionResponse:
    resolved_id = (document_id or "").strip() or _default_document_id(file)
    kwargs = dict(
        document_id=resolved_id,
        file_name=file.filename,
        mime_type=file.content_type,
        metadata=_metadata(metadata_json),
        operation_id=operation_id,
        include_trace=debug,
    )
    operation = module.knowledge.reindex if force else module.knowledge.ingest
    result = operation(file.file, **kwargs)
    if isinstance(result, list):
        raise RuntimeError("single-file HTTP ingestion unexpectedly returned multiple results")
    return ingestion_to_response(result, debug=debug)


@router.post("/documents", response_model=IngestionResponse, response_model_exclude_none=True)
def add_document(
    response: Response,
    file: Annotated[UploadFile, File(description="Document to ingest")],
    document_id: Annotated[str | None, Form()] = None,
    metadata_json: Annotated[str | None, Form()] = None,
    operation_id: Annotated[str | None, Form()] = None,
    debug: Annotated[bool, Query()] = False,
    module=Depends(get_rag_module),
) -> IngestionResponse:
    result = _ingest_file(
        module=module,
        file=file,
        document_id=document_id,
        metadata_json=metadata_json,
        operation_id=operation_id,
        debug=debug,
        force=False,
    )
    response.status_code = status.HTTP_201_CREATED if result.action == "add" else status.HTTP_200_OK
    return result


@router.put("/documents/{document_id}", response_model=IngestionResponse, response_model_exclude_none=True)
def update_document(
    document_id: str,
    file: Annotated[UploadFile, File(description="Replacement document")],
    metadata_json: Annotated[str | None, Form()] = None,
    operation_id: Annotated[str | None, Form()] = None,
    debug: Annotated[bool, Query()] = False,
    module=Depends(get_rag_module),
) -> IngestionResponse:
    return _ingest_file(
        module=module,
        file=file,
        document_id=document_id,
        metadata_json=metadata_json,
        operation_id=operation_id,
        debug=debug,
        force=False,
    )


@router.post("/documents/{document_id}/reindex", response_model=IngestionResponse, response_model_exclude_none=True)
def reindex_document(
    document_id: str,
    file: Annotated[UploadFile, File(description="Source used for forced reindex")],
    metadata_json: Annotated[str | None, Form()] = None,
    operation_id: Annotated[str | None, Form()] = None,
    debug: Annotated[bool, Query()] = False,
    module=Depends(get_rag_module),
) -> IngestionResponse:
    return _ingest_file(
        module=module,
        file=file,
        document_id=document_id,
        metadata_json=metadata_json,
        operation_id=operation_id,
        debug=debug,
        force=True,
    )


@router.post("/text", response_model=IngestionResponse, response_model_exclude_none=True)
def add_text(
    request: TextIngestionRequest,
    response: Response,
    module=Depends(get_rag_module),
) -> IngestionResponse:
    limit = int(module.config.parser.max_file_bytes)
    if len(request.text.encode("utf-8")) > limit:
        raise SourceTooLargeError(
            f"text source exceeds configured limit of {limit} bytes"
        )
    result = module.knowledge.ingest(
        request.text,
        source_type="text",
        document_id=request.document_id,
        metadata=request.metadata,
        operation_id=request.operation_id,
        include_trace=request.debug,
    )
    if isinstance(result, list):
        raise RuntimeError("single-text HTTP ingestion unexpectedly returned multiple results")
    payload = ingestion_to_response(result, debug=request.debug)
    response.status_code = status.HTTP_201_CREATED if payload.action == "add" else status.HTTP_200_OK
    return payload


@router.get("/documents", response_model=KnowledgeListResponse, response_model_exclude_none=True)
def list_documents(module=Depends(get_rag_module)) -> KnowledgeListResponse:
    return knowledge_list_to_response(
        module.knowledge.inspect(include_chunk_counts=True)
    )


@router.get("/documents/{document_id}", response_model=KnowledgeDetailResponse, response_model_exclude_none=True)
def get_document(document_id: str, module=Depends(get_rag_module)) -> KnowledgeDetailResponse:
    result = knowledge_detail_to_response(
        module.knowledge.inspect(document_id, include_chunk_counts=True), include_chunks=False
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return result


@router.get("/documents/{document_id}/chunks", response_model=KnowledgeDetailResponse, response_model_exclude_none=True)
def get_document_chunks(
    document_id: str,
    module=Depends(get_rag_module),
) -> KnowledgeDetailResponse:
    result = knowledge_detail_to_response(
        module.knowledge.inspect(document_id, include_chunks=True),
        include_chunks=True,
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return result


@router.delete("/documents/{document_id}", response_model=DeleteResponse, response_model_exclude_none=True)
def delete_document(
    document_id: str,
    operation_id: str | None = Query(default=None),
    module=Depends(get_rag_module),
) -> DeleteResponse:
    existing = module.knowledge.inspect(document_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    value = module.knowledge.delete(document_id, operation_id=operation_id)
    return delete_to_response(document_id, value)


@router.post("/recover", response_model=RecoveryResponse)
def recover(module=Depends(get_rag_module)) -> RecoveryResponse:
    return RecoveryResponse(recovered_operations=list(module.knowledge.recover()))
