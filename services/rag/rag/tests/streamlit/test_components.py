from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import pytest

from rag.config import RagConfig
from rag.contracts import DocumentSource, MetadataFilter
from rag.service import RagModule
from rag.streamlit_app.components.tab_ingestion import (
    execute_knowledge_operation,
    source_from_form,
)
from rag.streamlit_app.components.tab_inspection import inspect_knowledge
from rag.streamlit_app.components.tab_retrieval import parse_filters, retrieve_with_trace
from rag.streamlit_app.utils import get_rag_module, public_runtime_snapshot


@dataclass
class UploadedFile:
    name: str = "guide.pdf"
    type: str = "application/pdf"
    payload: bytes = b"%PDF-test"

    def getvalue(self):
        return self.payload


def test_source_form_returns_public_document_source_for_text_and_bytes():
    text_source = source_from_form("Text", "guide", None, "# Knowledge")
    assert text_source == DocumentSource(
        value="# Knowledge",
        source_type="text",
        metadata={"document_id": "guide", "ui_source": "streamlit"},
        file_name="guide.md",
        mime_type="text/markdown",
    )
    file_source = source_from_form("File", "pdf", UploadedFile(), "")
    assert file_source.source_type == "bytes"
    assert file_source.value == b"%PDF-test"
    assert file_source.file_name == "guide.pdf"
    assert file_source.mime_type == "application/pdf"


@pytest.mark.parametrize(
    ("source_kind", "document_id", "uploaded", "text", "message"),
    [
        ("Text", "", None, "content", "Document ID"),
        ("Text", "doc", None, "  ", "Text content"),
        ("File", "doc", None, "", "Select a file"),
    ],
)
def test_source_form_rejects_invalid_ui_intake(
    source_kind, document_id, uploaded, text, message
):
    with pytest.raises(ValueError, match=message):
        source_from_form(source_kind, document_id, uploaded, text)


def test_filter_parser_uses_public_metadata_filter_contracts():
    filters = parse_filters(
        '[{"field":"language","operator":"=","value":"vi"},'
        '{"field":"page","operator":">=","value":2}]'
    )
    assert filters == [
        MetadataFilter("language", "=", "vi"),
        MetadataFilter("page", ">=", 2),
    ]
    assert parse_filters("  ") is None


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '["not-an-object"]',
        '[{"field":"x"}]',
        '[{"field":"x","operator":"unknown","value":1}]',
    ],
)
def test_filter_parser_reports_invalid_filters_before_public_call(raw):
    with pytest.raises((ValueError, KeyError)):
        parse_filters(raw)


def test_inspection_requests_active_chunks_only_through_public_option():
    class Knowledge:
        def __init__(self):
            self.calls = []

        def inspect(self, document_id=None, *, include_chunks=False):
            self.calls.append((document_id, include_chunks))
            return {"documents": [], "active_chunks": []}

    knowledge = Knowledge()
    result = inspect_knowledge(knowledge, "doc")
    assert result == {"documents": [], "active_chunks": []}
    assert knowledge.calls == [("doc", True)]


def test_inspection_remains_compatible_with_public_facade_without_chunk_option():
    class Knowledge:
        def inspect(self, document_id=None):
            return {"document_id": document_id, "status": "active"}

    assert inspect_knowledge(Knowledge(), "doc")["document_id"] == "doc"


def test_ui_requests_bounded_public_traces_only_when_facade_supports_them():
    calls = []

    class Knowledge:
        def ingest(self, source, *, include_trace=False):
            calls.append(("ingest", include_trace))
            return "ingested"

        def update(self, source, *, include_trace=False):
            calls.append(("update", include_trace))
            return "updated"

        def reindex(self, source, *, include_trace=False):
            calls.append(("reindex", include_trace))
            return "reindexed"

    source = DocumentSource("knowledge", source_type="text")
    assert execute_knowledge_operation(Knowledge(), "Update", source) == "updated"

    class Module:
        def retrieve(self, query, *, include_trace=False, **kwargs):
            calls.append(("retrieve", include_trace))
            return "retrieved"

    assert retrieve_with_trace(Module(), "query", mode="hybrid") == "retrieved"
    assert calls == [("update", True), ("retrieve", True)]


def test_public_facade_can_be_constructed_for_streamlit_without_private_store_access():
    class PresentationPipeline:
        config = RagConfig()

        def inspect(self, document_id=None, *, include_chunks=False):
            return [] if document_id is None else None

        def runtime_snapshot(self):
            return {"configured": self.config.snapshot(), "resolved": {}}

        def recover(self):
            return []

        def close(self):
            self.closed = True

    pipeline = PresentationPipeline()
    module = RagModule(pipeline=pipeline)
    assert module.config is pipeline.config
    assert module.knowledge.inspect() == []
    snapshot = public_runtime_snapshot(module)
    assert snapshot["configured"] == pipeline.config.snapshot()
    module.close()
    assert pipeline.closed is True


def test_get_rag_module_cache_input_is_exact_effective_config(monkeypatch):
    captured = []
    sentinel = object()

    def fake_loader(key, config):
        captured.append((key, config))
        return sentinel

    monkeypatch.setattr("rag.streamlit_app.utils._load_rag_module", fake_loader)
    config = RagConfig()
    assert get_rag_module(config) is sentinel
    assert captured[0][1] is config
    from rag.streamlit_app.utils import config_fingerprint

    assert captured[0][0] == config_fingerprint(config)


def test_streamlit_resource_cache_reuses_only_the_exact_effective_config(monkeypatch):
    import rag.streamlit_app.utils as utils

    created = []

    class FakeRagModule:
        def __init__(self, config):
            self.config = config
            created.append(self)

    utils._load_rag_module.clear()
    monkeypatch.setattr(utils, "RagModule", FakeRagModule)
    first_config = RagConfig()
    second_config = replace(
        first_config,
        retrieval=replace(
            first_config.retrieval,
            candidate_k=first_config.retrieval.candidate_k + 1,
        ),
    )
    try:
        first = utils.get_rag_module(first_config)
        assert utils.get_rag_module(first_config) is first
        second = utils.get_rag_module(second_config)
        assert second is not first
        assert [item.config for item in created] == [first_config, second_config]
    finally:
        utils._load_rag_module.clear()


def test_runtime_snapshot_rejects_stale_or_mismatched_config():
    class Module:
        config = RagConfig()

        def runtime_snapshot(self):
            configured = self.config.snapshot()
            configured["retrieval"]["candidate_k"] += 1
            return {"configured": configured, "resolved": {}}

    with pytest.raises(RuntimeError, match="different configured value"):
        public_runtime_snapshot(Module())
