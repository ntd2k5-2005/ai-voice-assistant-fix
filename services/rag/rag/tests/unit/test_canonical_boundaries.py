from rag.contracts import DocumentSource, MetadataFilter
from rag.ingestion.parsers import (
    CSVParser,
    DOCXParser,
    HTMLParser,
    MarkdownParser,
    PDFParser,
    PPTXParser,
    TextParser,
    XLSXParser,
    default_registry,
)


def test_format_parsers_are_real_canonical_dispatch_targets():
    assert isinstance(default_registry.for_source(DocumentSource("hello", "text")), TextParser)
    assert isinstance(default_registry._file._parsers["md"], MarkdownParser)
    assert isinstance(default_registry._file._parsers["pdf"], PDFParser)
    assert isinstance(default_registry._file._parsers["docx"], DOCXParser)
    assert isinstance(default_registry._file._parsers["csv"], CSVParser)
    assert isinstance(default_registry._file._parsers["xlsx"], XLSXParser)
    assert isinstance(default_registry._file._parsers["pptx"], PPTXParser)
    assert isinstance(default_registry._file._parsers["html"], HTMLParser)


def test_metadata_filter_fails_closed_for_incompatible_types():
    assert not MetadataFilter("count", ">", "not-a-number").matches({"count": 1})
    assert not MetadataFilter("tags", "contains", "x").matches({"tags": 1})


def test_pipeline_normalizes_query_before_canonical_query_embedding():
    from rag.pipeline import RetrievalPipeline

    class Embedder:
        def __init__(self):
            self.queries = []

        def embed_query(self, text):
            self.queries.append(text)
            return [1.0, 0.0]

    class Store:
        def search(self, vector, top_k=10, query_filter=None):
            return []

    embedder = Embedder()
    RetrievalPipeline(embedder, Store()).retrieve_result(
        "  normalized query  ", mode="dense"
    )
    assert embedder.queries == ["normalized query"]
