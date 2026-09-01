from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from urllib.parse import quote
from typing import Any

import httpx

from ..contracts import RAGContext, RAGLatency, RAGResult, ServiceStatus
from ..errors import ContractError, ServiceError
from .base import clean_base_url, expect_json_object, response_error


class RAGClient:
    service_name = "RAG"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        require_rerank_applied: bool = True,
    ) -> None:
        self.base_url = clean_base_url(base_url)
        self.require_rerank_applied = require_rerank_applied
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_s))

    def close(self) -> None:
        self.client.close()

    def health(self) -> ServiceStatus:
        started = time.perf_counter()
        try:
            r = self.client.get(f"{self.base_url}/health/ready")
            latency = (time.perf_counter() - started) * 1000
            if r.status_code != 200:
                return ServiceStatus(
                    service=self.service_name,
                    reachable=True,
                    ready=False,
                    latency_ms=latency,
                    message=f"HTTP {r.status_code}",
                )
            data = expect_json_object(self.service_name, r)
            return ServiceStatus(
                service=self.service_name,
                reachable=True,
                ready=data.get("status") == "ready",
                latency_ms=latency,
                message="RAG service ready",
                details=data,
            )
        except Exception as exc:
            return ServiceStatus(
                service=self.service_name,
                reachable=False,
                ready=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                message=str(exc),
            )

    def retrieve(
        self,
        query: str,
        *,
        mode: str,
        rerank: bool,
        top_k: int,
        candidate_k: int,
        filters: list[dict[str, Any]] | None = None,
        debug: bool = False,
    ) -> RAGResult:
        if not query.strip():
            raise ServiceError(self.service_name, "Query is empty")
        if candidate_k < top_k:
            raise ServiceError(
                self.service_name,
                "candidate_k must be >= top_k",
            )

        payload = {
            "query": query.strip(),
            "mode": mode,
            "rerank": bool(rerank),
            "top_k": int(top_k),
            "candidate_k": int(candidate_k),
            "filters": filters or [],
            "debug": bool(debug),
        }

        started = time.perf_counter()
        try:
            r = self.client.post(
                f"{self.base_url}/v1/retrieval",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise ServiceError(
                self.service_name,
                "Retrieval timed out",
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(
                self.service_name,
                f"HTTP error: {exc}",
            ) from exc

        client_ms = (time.perf_counter() - started) * 1000
        if r.status_code != 200:
            raise response_error(self.service_name, r)

        body = expect_json_object(self.service_name, r)
        required = {
            "query",
            "mode",
            "rerank_requested",
            "rerank_applied",
            "contexts",
            "latency",
        }
        missing = required - set(body)
        if missing:
            raise ContractError(
                f"RAG: response missing fields: {sorted(missing)}"
            )

        if (
            rerank
            and self.require_rerank_applied
            and not bool(body.get("rerank_applied"))
        ):
            raise ServiceError(
                self.service_name,
                "Reranker requested but rerank_applied=false",
                details=body.get("diagnostics"),
            )

        contexts = [
            RAGContext.model_validate(item)
            for item in body["contexts"]
        ]
        latency = RAGLatency.model_validate(body["latency"])

        return RAGResult(
            query=body["query"],
            mode=body["mode"],
            rerank_requested=bool(body["rerank_requested"]),
            rerank_applied=bool(body["rerank_applied"]),
            contexts=contexts,
            latency=latency,
            client_latency_ms=client_ms,
        )

    def list_documents(self) -> list[dict[str, Any]]:
        r = self.client.get(
            f"{self.base_url}/v1/knowledge/documents"
        )
        if r.status_code != 200:
            raise response_error(self.service_name, r)
        body = expect_json_object(self.service_name, r)
        docs = body.get("documents")
        if not isinstance(docs, list):
            raise ContractError(
                "RAG: documents response missing list field 'documents'"
            )
        return docs

    def get_document(
        self,
        document_id: str,
        *,
        include_chunks: bool = False,
    ) -> dict[str, Any]:
        document_id = document_id.strip()
        if not document_id:
            raise ServiceError(
                self.service_name,
                "document_id is empty",
            )
        suffix = "/chunks" if include_chunks else ""
        encoded_id = quote(document_id, safe="")
        r = self.client.get(
            f"{self.base_url}/v1/knowledge/documents/"
            f"{encoded_id}{suffix}"
        )
        if r.status_code != 200:
            raise response_error(self.service_name, r)
        return expect_json_object(self.service_name, r)

    def upload_document(
        self,
        file_path: str | Path,
        *,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        reindex: bool = False,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise ServiceError(
                self.service_name,
                f"Knowledge file not found: {path}",
            )

        mime = (
            mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        data: dict[str, str] = {}
        if metadata:
            data["metadata_json"] = json.dumps(
                metadata,
                ensure_ascii=False,
            )

        if reindex:
            if not document_id or not document_id.strip():
                raise ServiceError(
                    self.service_name,
                    "document_id is required for reindex",
                )
            encoded_id = quote(document_id.strip(), safe="")
            url = (
                f"{self.base_url}/v1/knowledge/documents/"
                f"{encoded_id}/reindex"
            )
        else:
            url = f"{self.base_url}/v1/knowledge/documents"
            if document_id and document_id.strip():
                data["document_id"] = document_id.strip()

        with path.open("rb") as fh:
            r = self.client.post(
                url,
                files={
                    "file": (
                        path.name,
                        fh,
                        mime,
                    )
                },
                data=data,
            )

        if r.status_code not in (200, 201):
            raise response_error(self.service_name, r)
        return expect_json_object(self.service_name, r)

    def delete_document(
        self,
        document_id: str,
    ) -> dict[str, Any]:
        document_id = document_id.strip()
        if not document_id:
            raise ServiceError(
                self.service_name,
                "document_id is empty",
            )
        encoded_id = quote(document_id, safe="")
        r = self.client.delete(
            f"{self.base_url}/v1/knowledge/documents/{encoded_id}"
        )
        if r.status_code != 200:
            raise response_error(self.service_name, r)
        return expect_json_object(self.service_name, r)
