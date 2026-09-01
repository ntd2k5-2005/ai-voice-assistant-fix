from ...contracts import DocumentSource, ParsedDocument

from .base import (
    DocumentParser,
    ParserError,
    ParserErrorCode,
    one_file_document,
    read_text_file,
    source_path,
)


class TextParser(DocumentParser):
    """Parse an already available text value (not a filesystem path)."""

    def parse(self, source: DocumentSource):
        if not isinstance(source.value, str) or not source.value.strip():
            raise ParserError(ParserErrorCode.EMPTY_CONTENT, "text source must not be empty")
        return [
            ParsedDocument(
                str(source.metadata.get("document_id") or "raw-text"),
                "raw-text",
                "text",
                source.value,
                metadata={
                    **dict(source.metadata),
                    "parser_method": "text",
                    "file_type": "text",
                    "source_name": "raw-text",
                    "segment_index": 0,
                    "block_type": "text",
                },
            )
        ]


class PlainTextFileParser(DocumentParser):
    """Decode a plain-text/code file while retaining line semantics."""

    def __init__(self, encoding_fallbacks=None):
        self.encoding_fallbacks = encoding_fallbacks

    def parse(self, source: DocumentSource):
        path = source_path(source)
        text, encoding = read_text_file(path, self.encoding_fallbacks)
        if not text.strip():
            raise ParserError(ParserErrorCode.EMPTY_CONTENT, "text file is empty", source=path)
        return one_file_document(
            source,
            path,
            text,
            "text-file",
            section="text",
            block_type="text",
            encoding=encoding,
            line_start=1,
            line_end=max(1, len(text.splitlines())),
        )
