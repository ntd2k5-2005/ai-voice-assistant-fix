"""Deterministic token- and structure-aware document chunking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping, Protocol

from ..config import ChunkingConfig
from ..errors import ChunkingError


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class RegexTokenCounter:
    """Dependency-free conservative counter used when no model tokenizer is set.

    CJK characters and punctuation count individually. Alphanumeric runs count
    in four-character pieces, avoiding the dangerous ``words * 1.3`` estimate
    for long unbroken strings while remaining deterministic and inexpensive.
    """

    _atoms = re.compile(
        r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]"
        r"|[^\W_]+|_+|[^\w\s]",
        re.UNICODE,
    )

    def count(self, text: str) -> int:
        total = 0
        for match in self._atoms.finditer(text):
            atom = match.group(0)
            if len(atom) == 1 and (not atom.isalnum() or ord(atom) >= 0x3040):
                total += 1
            else:
                total += max(1, math.ceil(len(atom) / 4))
        return total


class ModelTokenCounter:
    """Adapter for HuggingFace-compatible tokenizers."""

    def __init__(self, tokenizer) -> None:
        if not hasattr(tokenizer, "encode"):
            raise TypeError("tokenizer must provide encode()")
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            tokens = self.tokenizer.encode(text)
        return len(tokens)


class CallableTokenCounter:
    def __init__(self, callback: Callable[[str], int]) -> None:
        self.callback = callback

    def count(self, text: str) -> int:
        value = int(self.callback(text))
        if value < 0:
            raise ChunkingError("token counter returned a negative value")
        return value


@dataclass(frozen=True)
class LogicalUnit:
    text: str
    heading: str | None
    block_type: str
    logical_index: int
    metadata: Mapping[str, Any]


_MARKDOWN_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")
_NAMED_HEADING = re.compile(
    r"^\s*(?:(?:chương|mục|phần|bài|câu|chapter|section)\s+[\w.-]+(?:\s+.+)?|\d+(?:\.\d+)*[.)]?\s+\S.+)$",
    re.IGNORECASE,
)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*•]|\d+[.)])\s+")
_BOUNDARY_PATTERNS = (
    re.compile(r"\n\n+"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?;:…])\s+"),
    re.compile(r"\s+"),
)

# Conservative structure cues used only when a parser had to preserve several
# semantic sub-sections inside one text unit. Parser-emitted structure remains
# authoritative; these cues recover common report/document subheadings without
# treating ordinary sentences as headings.
_STRUCTURAL_SUBHEADING = re.compile(
    r"^(?:"
    r"từ\s+|đối\s+với\s+|trong\s+(?:học tập|công việc|thực tế|nhà trường|gia đình)|"
    r"tiểu\s+kết$|kết\s+luận$|bối\s+cảnh$|khái\s+niệm$|biểu\s+hiện$|"
    r"nguyên\s+nhân$|hậu\s+quả$|đánh\s+giá$|mục\s+tiêu|lý\s+do|"
    r"giải\s+pháp|biện\s+pháp|phương\s+pháp|kết\s+quả|sản\s+phẩm|"
    r"nhìn\s+nhận|liên\s+hệ|"
    r"background$|concepts?$|manifestations?$|causes?$|effects?$|solutions?$|"
    r"for\s+|from\s+|conclusion$|summary$"
    r")",
    re.IGNORECASE,
)
_SENTENCE_TERMINATOR = re.compile(r"[.!?…;]$", re.UNICODE)


class UltimateChunker:
    """Chunk parsed units within exact token and provenance boundaries.

    Explicit arguments override ``ChunkingConfig``. ``model_max_tokens`` is
    used only when config leaves ``max_tokens`` on auto; configured reserved
    tokens are then subtracted from the resolved model/fallback limit.

    ``max_chars``/``overlap_chars`` remain accepted for old local UI callers,
    but they now select token budgets instead of re-enabling character slicing.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        *,
        reserved_tokens: int | None = None,
        model_max_tokens: int | None = None,
        token_counter: TokenCounter | Callable[[str], int] | None = None,
        tokenizer=None,
        tokenizer_name: str | None = None,
        config: ChunkingConfig | None = None,
        include_heading: bool = True,
        max_chars: int | None = None,
        overlap_chars: int | None = None,
        hard_max_chars: int | None = None,
    ) -> None:
        config = config or ChunkingConfig()
        legacy = max_chars is not None
        if legacy:
            max_tokens = int(max_chars)
            overlap_tokens = int(overlap_chars or 0)
            reserved_tokens = 0
            if hard_max_chars is not None and not (0 < max_chars <= hard_max_chars):
                raise ValueError("Require 0 < max_chars <= hard_max_chars")
        resolved_limit = (
            int(max_tokens)
            if max_tokens is not None
            else int(config.max_tokens)
            if config.max_tokens is not None
            else int(model_max_tokens)
            if model_max_tokens is not None
            else int(config.fallback_max_tokens)
        )
        reserved = int(config.reserved_tokens if reserved_tokens is None else reserved_tokens)
        self.max_tokens = resolved_limit
        self.reserved_tokens = reserved
        self.effective_max_tokens = resolved_limit - reserved
        self.overlap_tokens = int(config.overlap_tokens if overlap_tokens is None else overlap_tokens)
        self.min_chunk_tokens = int(config.min_chunk_tokens)
        self.max_chunks_per_document = int(config.max_chunks_per_document)
        self.preserve_tables = bool(config.preserve_tables)
        self.include_heading = bool(include_heading)
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.hard_max_chars = hard_max_chars

        if self.effective_max_tokens <= 0:
            raise ValueError("reserved_tokens must be smaller than the resolved token limit")
        if not 0 <= self.overlap_tokens < self.effective_max_tokens:
            raise ValueError("Require 0 <= overlap_tokens < effective_max_tokens")

        if token_counter is not None:
            self.counter = (
                CallableTokenCounter(token_counter)
                if callable(token_counter) and not hasattr(token_counter, "count")
                else token_counter
            )
        elif tokenizer is not None:
            self.counter = ModelTokenCounter(tokenizer)
        else:
            selected_name = tokenizer_name or config.tokenizer_name
            if selected_name:
                try:
                    from transformers import AutoTokenizer

                    loaded = AutoTokenizer.from_pretrained(
                        selected_name,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                except Exception as exc:
                    raise ChunkingError(
                        f"configured tokenizer {selected_name!r} could not be loaded locally: {exc}"
                    ) from exc
                self.counter = ModelTokenCounter(loaded)
            else:
                self.counter = RegexTokenCounter()

    def _estimate_tokens(self, text: str) -> int:
        """Compatibility name backed by the configured token counter."""

        return self._count(text)

    def _count(self, text: str) -> int:
        return int(self.counter.count(text))

    @staticmethod
    def _clean_heading(value: Any) -> str:
        text = str(value or "").strip().strip("#").strip().strip("[]").strip()
        return text.rstrip(":").strip()

    def _is_probable_subheading(self, line: str, next_line: str | None) -> bool:
        value = self._clean_heading(line)
        if not value or not any(char.isalpha() for char in value):
            return False
        if _LIST_ITEM.match(value) or _TABLE_ROW.match(value):
            return False
        if len(value) > 120 or self._count(value) > 18:
            return False
        if _SENTENCE_TERMINATOR.search(value):
            return False
        # A colon alone is not enough: ordinary lead-in sentences such as
        # "The objectives are:" must remain body text unless another strong
        # structural signal exists.
        if _STRUCTURAL_SUBHEADING.match(value):
            return True
        if self._count(value) <= 6 and "," not in value and next_line:
            following = next_line.strip()
            return bool(following) and self._count(following) >= max(10, self._count(value) * 2)
        return False

    def _split_internal_sections(
        self, text: str, primary_heading: str | None
    ) -> list[tuple[str | None, str]]:
        """Recover sub-sections inside one parser unit without flattening them.

        DOCX/PDF parsers can legitimately emit a whole logical section as one
        unit. This pass recognizes only conservative standalone subheading lines
        and keeps paragraph boundaries explicit. It never token-slices here;
        oversized bodies are handled later by the normal token chunker.
        """

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return []
        primary = self._clean_heading(primary_heading)
        if primary and self._clean_heading(lines[0]).casefold() == primary.casefold():
            lines = lines[1:]

        output: list[tuple[str | None, str]] = []
        current_heading: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            # Preserve structural separation between original paragraphs/lines.
            body = "\n\n".join(part.strip() for part in buffer if part.strip()).strip()
            if body:
                output.append((current_heading, body))
            buffer = []

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            next_line = next(
                (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()),
                None,
            )
            if self._is_probable_subheading(line, next_line):
                flush()
                current_heading = self._clean_heading(line)
                continue
            buffer.append(line)
        flush()
        return output

    @staticmethod
    def _heading_is_child(parent: str, child: str) -> bool:
        """Conservatively infer a parent/child relation between headings.

        The parser's explicit heading levels remain preferable. This fallback
        exists for common internal reports whose visual formatting makes every
        heading look like level 1. It intentionally prefers false negatives
        over inventing a hierarchy between unrelated headings.
        """

        p = parent.casefold().strip()
        c = child.casefold().strip()
        if not p or not c or p == c:
            return False
        if c.startswith(("đối với ", "từ ", "trong ", "for ", "from ")):
            if any(key in p for key in ("giải pháp", "biện pháp", "nguyên nhân", "hậu quả", "biểu hiện", "solution", "cause", "effect", "manifest")):
                return True
        relations = {
            "đặt vấn đề": ("bối cảnh", "lý do", "mục tiêu"),
            "thực trạng": ("khái niệm", "biểu hiện", "nguyên nhân", "hậu quả", "đánh giá"),
            "đánh giá": ("nhìn nhận", "liên hệ", "tiểu kết", "kết luận"),
            "giải pháp": ("đối với", "một số lưu ý"),
            "biện pháp": ("đối với", "một số lưu ý"),
            "background": ("context", "reason", "objective"),
            "solutions": ("for ", "notes"),
        }
        for parent_key, child_prefixes in relations.items():
            if parent_key in p and any(c.startswith(prefix) for prefix in child_prefixes):
                return True
        return False

    @staticmethod
    def _metadata_heading_path(metadata: Mapping[str, Any]) -> list[str]:
        raw = metadata.get("heading_path")
        if not isinstance(raw, (list, tuple)):
            return []
        result = []
        for item in raw:
            value = str(item or "").strip().rstrip(":").strip()
            if value and (not result or result[-1].casefold() != value.casefold()):
                result.append(value)
        return result

    def _is_heading_only_document(self, doc: Mapping[str, Any]) -> str | None:
        metadata = dict(doc.get("metadata") or {})
        text = str(doc.get("text") or "").strip()
        heading = self._clean_heading(
            metadata.get("heading") or metadata.get("section") or doc.get("heading")
        )
        if not text or not heading:
            return None
        block_type = str(metadata.get("block_type") or doc.get("block_type") or "").casefold()
        text_clean = self._clean_heading(text)
        same = text_clean.casefold() == heading.casefold()
        # Exact short single-line equality is itself a strong structural signal
        # even when an observability/adapter layer omitted block_type metadata.
        if (
            same
            and "\n" not in text.strip()
            and self._count(text) <= 24
            and not _SENTENCE_TERMINATOR.search(text_clean)
        ):
            return heading
        if same and block_type in {"heading", "section", "title"} and self._count(text) <= 24:
            return heading
        return None

    def _chunk_body_with_path(
        self, doc: Mapping[str, Any], body: str, heading_path: list[str], block_type: str
    ) -> list[dict[str, Any]]:
        if not body.strip():
            return []
        clone = dict(doc)
        clone_metadata = dict(doc.get("metadata") or {})
        context_path = [self._clean_heading(item) for item in heading_path if self._clean_heading(item)]
        context_label = " > ".join(context_path)
        clone["text"] = body.strip()
        clone["metadata"] = clone_metadata
        # A compact full path is prepended to embedding text by the existing
        # renderer. Metadata is patched back to leaf/path semantics below.
        if context_label:
            clone["heading"] = context_label
            clone_metadata["heading"] = context_label
            clone_metadata["section"] = context_label
        clone_metadata["block_type"] = block_type
        chunks = self.chunk_document(clone)
        for chunk in chunks:
            metadata = chunk["metadata"]
            if context_path:
                metadata["heading"] = context_path[-1]
                metadata["heading_path"] = list(context_path)
                metadata["section"] = context_label
                metadata["parent_section"] = context_path[-2] if len(context_path) > 1 else None
                metadata["section_depth"] = len(context_path)
            metadata["block_type"] = block_type
        return chunks

    def chunk_documents(self, documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Chunk an ordered parser-unit sequence with cross-unit structure.

        RAGFlow-inspired principle: headings are structural context, not useful
        standalone knowledge. Short heading-only parser units are deferred and
        attached to following body/subsection chunks. Internal subheadings are
        recovered conservatively, and hierarchy is prepended to embedding text
        while also remaining typed metadata.
        """

        docs = [dict(item) for item in documents]
        chunks: list[dict[str, Any]] = []
        # stack entries: {text, has_content}. A standalone heading following a
        # consumed root starts a new root; consecutive standalone headings form
        # a parent/child path until content arrives.
        stack: list[dict[str, Any]] = []

        for doc in docs:
            metadata = dict(doc.get("metadata") or {})
            doc["metadata"] = metadata
            text = str(doc.get("text") or "")
            if not text.strip():
                continue

            heading_only = self._is_heading_only_document(doc)
            if heading_only:
                if stack and stack[-1]["text"].casefold() == heading_only.casefold():
                    continue
                if not stack or bool(stack[0].get("has_content")):
                    stack = [{"text": heading_only, "has_content": False}]
                elif self._heading_is_child(str(stack[-1]["text"]), heading_only):
                    stack.append({"text": heading_only, "has_content": False})
                else:
                    # Consecutive unrelated title/heading labels are not
                    # automatically nested. Keep only the most recent pending
                    # structural label; if it has no body it should not become
                    # searchable knowledge by itself.
                    stack = [{"text": heading_only, "has_content": False}]
                continue

            unit_heading = self._clean_heading(
                metadata.get("heading") or metadata.get("section") or doc.get("heading")
            )
            explicit_path = self._metadata_heading_path(metadata)

            # Parser-provided multi-level hierarchy is authoritative. This is
            # how Markdown/HTML and properly styled DOCX preserve exact sibling
            # paths; lexical fallback below is only for documents whose parser
            # can recover labels but not trustworthy levels.
            if len(explicit_path) > 1:
                base_path = explicit_path
                stack = [{"text": value, "has_content": False} for value in base_path]
            elif stack:
                if unit_heading:
                    current_heading = str(stack[-1]["text"])
                    if current_heading.casefold() == unit_heading.casefold():
                        pass
                    elif not bool(stack[-1].get("has_content")):
                        if self._heading_is_child(current_heading, unit_heading):
                            stack.append({"text": unit_heading, "has_content": False})
                        else:
                            stack = [{"text": unit_heading, "has_content": False}]
                    elif (
                        len(stack) > 1
                        and self._heading_is_child(str(stack[-2]["text"]), unit_heading)
                    ):
                        stack[-1] = {"text": unit_heading, "has_content": False}
                    elif self._heading_is_child(current_heading, unit_heading):
                        stack.append({"text": unit_heading, "has_content": False})
                    else:
                        stack = [{"text": unit_heading, "has_content": False}]
                base_path = [str(entry["text"]) for entry in stack]
            elif unit_heading:
                base_path = [unit_heading]
                stack = [{"text": unit_heading, "has_content": False}]
            else:
                base_path = []

            block_type = str(metadata.get("block_type") or doc.get("block_type") or "paragraph")
            structural_family = self._structural_family(metadata)
            created: list[dict[str, Any]] = []
            if structural_family in {"table", "code", "json_record", "notes"}:
                created.extend(self._chunk_body_with_path(doc, text, base_path, block_type))
            else:
                sections = self._split_internal_sections(text, unit_heading)
                if not sections:
                    sections = [(None, text.strip())]
                for subsection, body in sections:
                    path = list(base_path)
                    if subsection and (not path or path[-1].casefold() != subsection.casefold()):
                        path.append(subsection)
                    created.extend(self._chunk_body_with_path(doc, body, path, block_type))

            if created:
                chunks.extend(created)
                for entry in stack:
                    entry["has_content"] = True

        chunks = self.finalize_document_chunks(chunks)
        for index, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_index"] = index
        return chunks

    def chunk_document(self, doc: Mapping[str, Any]) -> list[dict[str, Any]]:
        text = str(doc.get("text") or "")
        if not text.strip():
            return []
        units = self._logical_units(text, doc)
        chunks: list[dict[str, Any]] = []

        group: list[LogicalUnit] = []
        group_key = None

        def flush_group() -> None:
            nonlocal group
            if not group:
                return
            raw = "\n\n".join(unit.text for unit in group if unit.text.strip())
            if raw:
                block_types = {unit.block_type for unit in group}
                block_type = next(iter(block_types)) if len(block_types) == 1 else "text"
                pieces = self._split_raw(raw, group[0].heading, block_type)
                for piece, overlap_count in pieces:
                    chunks.append(
                        self._create_payload(
                            piece,
                            doc,
                            len(chunks),
                            group[0].heading,
                            block_type,
                            group[0].logical_index,
                            group[-1].logical_index,
                            overlap_count,
                        )
                    )
            group = []

        for unit in units:
            key = self._unit_boundary_key(unit, doc)
            is_table = unit.block_type in {"table", "table_row", "table_header"} or self._is_markdown_table(unit.text)
            if is_table and self.preserve_tables:
                flush_group()
                for piece, table_metadata in self._split_table(unit.text, unit.heading):
                    chunks.append(
                        self._create_payload(
                            piece,
                            doc,
                            len(chunks),
                            unit.heading,
                            unit.block_type,
                            unit.logical_index,
                            unit.logical_index,
                            int(table_metadata.pop("overlap_tokens", 0)),
                            table_metadata,
                        )
                    )
                group_key = None
                continue
            if group and key != group_key:
                flush_group()
            group_key = key
            group.append(unit)
        flush_group()

        chunks = self.finalize_document_chunks(chunks)
        if len(chunks) > self.max_chunks_per_document:
            raise ChunkingError(
                f"document produced {len(chunks)} chunks, exceeding configured limit "
                f"{self.max_chunks_per_document}"
            )
        for chunk in chunks:
            count = self._count(chunk["text"])
            if count > self.effective_max_tokens:
                raise ChunkingError(
                    f"chunk {chunk['chunk_id']} has {count} tokens, limit is {self.effective_max_tokens}"
                )
        return chunks

    @staticmethod
    def _structural_family(metadata: Mapping[str, Any]) -> str:
        block_type = str(metadata.get("block_type") or "text")
        parser_method = str(metadata.get("parser_method") or metadata.get("method") or "").casefold()
        if (
            metadata.get("table_id") is not None
            or metadata.get("table_index") is not None
            or block_type.startswith("table")
            or parser_method.endswith("-table")
        ):
            return "table"
        if block_type == "code":
            return "code"
        if block_type in {"json_record", "notes"}:
            return block_type
        return "text"

    @classmethod
    def _compatible_short_merge(
        cls, left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> bool:
        left_metadata = dict(left.get("metadata") or {})
        right_metadata = dict(right.get("metadata") or {})
        for key in (
            "document_id",
            "version",
            "source",
            "page_number",
            "section",
            "sheet_name",
            "slide_number",
            "table_id",
        ):
            if left_metadata.get(key) != right_metadata.get(key):
                return False
        if cls._structural_family(left_metadata) != cls._structural_family(right_metadata):
            return False
        left_parts = left_metadata.get("source_parts") or [left_metadata.get("source_part")]
        right_parts = right_metadata.get("source_parts") or [right_metadata.get("source_part")]
        left_numbers = [int(value) for value in left_parts if value is not None]
        right_numbers = [int(value) for value in right_parts if value is not None]
        return not left_numbers or not right_numbers or min(right_numbers) <= max(left_numbers) + 1

    @staticmethod
    def _strip_exact_overlap(left: str, right: str) -> str:
        maximum = min(len(left), len(right))
        for size in range(maximum, 0, -1):
            if left[-size:] == right[:size]:
                return right[size:].lstrip()
        return right

    @classmethod
    def _join_adjacent_text(
        cls, left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> str:
        left_text = str(left.get("text") or "").rstrip()
        right_text = str(right.get("text") or "").lstrip()
        left_metadata = dict(left.get("metadata") or {})
        right_metadata = dict(right.get("metadata") or {})
        if cls._structural_family(left_metadata) == "table":
            left_lines = left_text.splitlines()
            right_lines = right_text.splitlines()
            if (
                len(left_lines) >= 2
                and len(right_lines) >= 2
                and left_lines[:2] == right_lines[:2]
            ):
                right_text = "\n".join(right_lines[2:]).lstrip()
        else:
            heading = right_metadata.get("heading")
            heading_prefix = f"[{heading}]\n" if heading else ""
            if heading_prefix and left_text.startswith(heading_prefix) and right_text.startswith(heading_prefix):
                right_text = right_text[len(heading_prefix):].lstrip()
            right_text = cls._strip_exact_overlap(left_text, right_text)
        if not left_text:
            return right_text
        if not right_text:
            return left_text
        return f"{left_text}\n\n{right_text}"

    def _merged_payload(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        text: str,
    ) -> dict[str, Any]:
        left_metadata = dict(left.get("metadata") or {})
        right_metadata = dict(right.get("metadata") or {})
        component_ids = [
            *left_metadata.get("component_chunk_ids", [str(left["chunk_id"])]),
            *right_metadata.get("component_chunk_ids", [str(right["chunk_id"])]),
        ]
        source_parts = sorted(
            {
                int(value)
                for value in (
                    *(left_metadata.get("source_parts") or [left_metadata.get("source_part")]),
                    *(right_metadata.get("source_parts") or [right_metadata.get("source_part")]),
                )
                if value is not None
            }
        )
        token_count = self._count(text)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        logical_start = min(
            int(left_metadata.get("logical_start", 0)),
            int(right_metadata.get("logical_start", 0)),
        )
        logical_end = max(
            int(left_metadata.get("logical_end", logical_start)),
            int(right_metadata.get("logical_end", logical_start)),
        )
        identity = json.dumps(
            {
                "document_id": str(left.get("document_id") or left_metadata.get("document_id")),
                "version": left_metadata.get("version", 1),
                "source_parts": source_parts,
                "logical_start": logical_start,
                "logical_end": logical_end,
                "component_chunk_ids": component_ids,
                "content_hash": content_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        metadata = {
            **left_metadata,
            "chunk_id": chunk_id,
            "content_hash": content_hash,
            "token_count": token_count,
            "token_estimate": token_count,
            "logical_start": logical_start,
            "logical_end": logical_end,
            "source_parts": source_parts,
            "component_chunk_ids": component_ids,
            "merged_short_chunks": len(component_ids),
            "below_min_tokens": token_count < self.min_chunk_tokens,
            "min_chunk_tokens": self.min_chunk_tokens,
        }
        if source_parts:
            metadata["source_part"] = source_parts[0]
            metadata["source_part_end"] = source_parts[-1]
        if left_metadata.get("row_start") is not None or right_metadata.get("row_start") is not None:
            row_starts = [
                int(value)
                for value in (left_metadata.get("row_start"), right_metadata.get("row_start"))
                if value is not None
            ]
            row_ends = [
                int(value)
                for value in (left_metadata.get("row_end"), right_metadata.get("row_end"))
                if value is not None
            ]
            if row_starts:
                metadata["row_start"] = min(row_starts)
            if row_ends:
                metadata["row_end"] = max(row_ends)
        return {
            "_id": chunk_id,
            "chunk_id": chunk_id,
            "document_id": str(left.get("document_id") or left_metadata.get("document_id")),
            "text": text,
            "metadata": metadata,
        }

    def finalize_document_chunks(
        self, chunks: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge undersized adjacent chunks without crossing hard boundaries.

        Content is never discarded.  If neither neighbor is structurally
        compatible or a merge would exceed the token budget, the short chunk
        is retained and marked explicitly in metadata.
        """

        result = [
            {**dict(chunk), "metadata": dict(chunk.get("metadata") or {})}
            for chunk in chunks
        ]
        if self.min_chunk_tokens <= 1 or len(result) < 2:
            for chunk in result:
                count = self._count(str(chunk.get("text") or ""))
                chunk["metadata"].setdefault("below_min_tokens", count < self.min_chunk_tokens)
                chunk["metadata"].setdefault("min_chunk_tokens", self.min_chunk_tokens)
            return result

        index = 0
        while index < len(result):
            current = result[index]
            current_count = self._count(str(current.get("text") or ""))
            if current_count >= self.min_chunk_tokens:
                current["metadata"]["below_min_tokens"] = False
                current["metadata"]["min_chunk_tokens"] = self.min_chunk_tokens
                index += 1
                continue

            merged = False
            for neighbor_index in (index - 1, index + 1):
                if neighbor_index < 0 or neighbor_index >= len(result):
                    continue
                left_index, right_index = sorted((index, neighbor_index))
                left, right = result[left_index], result[right_index]
                if not self._compatible_short_merge(left, right):
                    continue
                joined = self._join_adjacent_text(left, right)
                if self._count(joined) > self.effective_max_tokens:
                    continue
                result[left_index] = self._merged_payload(left, right, joined)
                del result[right_index]
                index = max(0, left_index - 1)
                merged = True
                break
            if not merged:
                current["metadata"]["below_min_tokens"] = True
                current["metadata"]["min_chunk_tokens"] = self.min_chunk_tokens
                index += 1

        return result

    def _logical_units(self, text: str, doc: Mapping[str, Any]) -> list[LogicalUnit]:
        inherited = dict(doc.get("metadata") or {})
        inherited_heading = inherited.get("heading") or doc.get("heading")
        inherited_type = str(inherited.get("block_type") or doc.get("block_type") or "")
        if inherited_type in {"table", "table_row", "table_header", "code", "notes", "json_record"}:
            return [LogicalUnit(text.strip(), inherited_heading, inherited_type, 0, inherited)]

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        units: list[LogicalUnit] = []
        buffer: list[str] = []
        current_heading = str(inherited_heading) if inherited_heading else None
        current_type = inherited_type or "paragraph"

        def flush() -> None:
            nonlocal buffer, current_type
            value = "\n".join(buffer).strip("\n")
            if value.strip():
                units.append(LogicalUnit(value, current_heading, current_type, len(units), inherited))
            buffer = []
            current_type = inherited_type or "paragraph"

        index = 0
        while index < len(lines):
            line = lines[index]
            markdown_heading = _MARKDOWN_HEADING.match(line)
            named_heading = (
                _NAMED_HEADING.match(line)
                if len(line) <= 180 and self._count(line) <= 24
                else None
            )
            if markdown_heading or named_heading:
                flush()  # old content retains the old heading
                current_heading = (
                    markdown_heading.group(2).strip()
                    if markdown_heading
                    else line.strip().rstrip(":")
                )
                buffer = [line]
                current_type = "heading"
                index += 1
                continue

            if _TABLE_ROW.match(line):
                flush()
                table_lines = [line]
                index += 1
                while index < len(lines) and _TABLE_ROW.match(lines[index]):
                    table_lines.append(lines[index])
                    index += 1
                units.append(LogicalUnit("\n".join(table_lines), current_heading, "table", len(units), inherited))
                continue

            if not line.strip():
                flush()
                index += 1
                continue

            line_type = "list" if _LIST_ITEM.match(line) else inherited_type or "paragraph"
            if buffer and current_type != line_type and {current_type, line_type} & {"list"}:
                flush()
            if not buffer:
                current_type = line_type
            buffer.append(line)
            index += 1
        flush()
        return units

    @staticmethod
    def _unit_boundary_key(unit: LogicalUnit, doc: Mapping[str, Any]):
        metadata = dict(doc.get("metadata") or {})
        return (
            unit.heading,
            metadata.get("page_number", metadata.get("page", doc.get("page"))),
            metadata.get("sheet_name"),
            metadata.get("slide_number"),
            metadata.get("table_id"),
        )

    @staticmethod
    def _is_markdown_table(text: str) -> bool:
        lines = [line for line in text.splitlines() if line.strip()]
        return len(lines) >= 2 and _TABLE_ROW.match(lines[0]) is not None and _TABLE_SEPARATOR.match(lines[1]) is not None

    def _render(self, raw: str, heading: str | None, block_type: str) -> str:
        raw = raw.strip()
        if not raw or not self.include_heading or not heading or block_type == "heading":
            return raw
        first = raw.splitlines()[0].strip().lstrip("#").strip().strip("[]")
        if first.casefold() == str(heading).strip().casefold():
            return raw
        prefix = f"[{heading}]\n"
        if self._count(prefix + raw[:1]) > self.effective_max_tokens:
            return raw
        return prefix + raw

    def _split_raw(self, raw: str, heading: str | None, block_type: str) -> list[tuple[str, int]]:
        raw = raw.strip()
        if not raw:
            return []
        if self._count(self._render(raw, heading, block_type)) <= self.effective_max_tokens:
            return [(raw, 0)]

        pieces: list[tuple[str, int]] = []
        position = 0
        previous_raw = ""
        while position < len(raw):
            overlap = self._tail_for_tokens(previous_raw, self.overlap_tokens) if previous_raw else ""
            remaining = raw[position:]
            separator = "\n\n" if overlap and remaining and not overlap.endswith(("\n", " ")) and not remaining.startswith(("\n", " ")) else ""
            candidate = overlap + separator + remaining
            prefix_len = len(overlap) + len(separator)
            consumed = self._largest_prefix(candidate, heading, block_type)
            if consumed <= prefix_len:
                overlap = ""
                separator = ""
                prefix_len = 0
                candidate = remaining
                consumed = self._largest_prefix(candidate, heading, block_type)
            if consumed <= 0:
                raise ChunkingError("token budget is too small to fit one source character")
            piece = candidate[:consumed].strip()
            if not piece:
                position += max(1, consumed - prefix_len)
                continue
            overlap_count = self._count(overlap) if overlap else 0
            pieces.append((piece, overlap_count))
            new_characters = consumed - prefix_len
            if new_characters <= 0:
                raise ChunkingError("overlap prevented chunker progress")
            position += new_characters
            previous_raw = piece
        return pieces

    def _largest_prefix(self, value: str, heading: str | None, block_type: str) -> int:
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            rendered = self._render(value[:middle], heading, block_type)
            if self._count(rendered) <= self.effective_max_tokens:
                low = middle
            else:
                high = middle - 1
        limit = low
        if limit >= len(value):
            return limit
        minimum = max(1, int(limit * 0.55))
        prefix = value[:limit]
        for pattern in _BOUNDARY_PATTERNS:
            matches = [match.end() for match in pattern.finditer(prefix) if match.end() >= minimum]
            if matches:
                return matches[-1]
        return limit

    def _tail_for_tokens(self, value: str, budget: int) -> str:
        """Return an overlap suffix beginning at the strongest safe boundary.

        Prefer paragraph, line, then sentence boundaries; only fall back to a
        word/token boundary when no semantic boundary can fit the budget. This
        avoids fragments such as ``"nhân phổ biến..."`` at chunk starts.
        """

        text = (value or "").strip()
        if not text or budget <= 0:
            return ""
        if self._count(text) <= budget:
            return text

        boundary_patterns = (
            re.compile(r"\n\n+"),
            re.compile(r"\n+"),
            re.compile(r"(?<=[.!?;…])\s+"),
        )
        for pattern in boundary_patterns:
            starts = [match.end() for match in pattern.finditer(text)]
            fitting = [start for start in starts if self._count(text[start:].strip()) <= budget]
            if fitting:
                # Earliest fitting boundary preserves as much complete semantic
                # context as the overlap budget permits.
                return text[min(fitting):].strip()

        # If semantic punctuation/line structure exists but no complete unit
        # fits the overlap budget, prefer *no overlap* over a sentence fragment.
        if "\n" in text or re.search(r"[.!?;…]", text):
            return ""

        # Last-resort word boundary for genuinely unstructured text. Never
        # start in the middle of a word.
        words = list(re.finditer(r"\S+(?:\s+|$)", text))
        for match in words:
            suffix = text[match.start():].strip()
            if self._count(suffix) <= budget:
                return suffix
        return ""

    def _split_table(self, text: str, heading: str | None):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not self._is_markdown_table(text):
            return [
                (piece, {"overlap_tokens": overlap, "table_split": len(pieces) > 1})
                for pieces in [self._split_raw(text, heading, "table_row")]
                for piece, overlap in pieces
            ]

        header, separator, rows = lines[0], lines[1], lines[2:]
        if not rows:
            pieces = self._split_raw("\n".join([header, separator]), heading, "table")
            return [(piece, {"row_start": 1, "row_end": 1, "table_split": len(pieces) > 1, "overlap_tokens": overlap}) for piece, overlap in pieces]

        output: list[tuple[str, dict[str, Any]]] = []
        current: list[str] = []
        start_row = 2
        for row_offset, row in enumerate(rows, 2):
            candidate_rows = current + [row]
            candidate = "\n".join([header, separator, *candidate_rows])
            if current and self._count(self._render(candidate, heading, "table")) > self.effective_max_tokens:
                value = "\n".join([header, separator, *current])
                output.append((value, {"row_start": start_row, "row_end": row_offset - 1, "table_split": True, "header_repeated": bool(output)}))
                current = []
                start_row = row_offset
            single = "\n".join([header, separator, row])
            if not current and self._count(self._render(single, heading, "table")) > self.effective_max_tokens:
                output.extend(self._split_oversized_table_row(header, separator, row, row_offset, heading))
                start_row = row_offset + 1
            else:
                current.append(row)
        if current:
            value = "\n".join([header, separator, *current])
            output.append((value, {"row_start": start_row, "row_end": start_row + len(current) - 1, "table_split": bool(output), "header_repeated": bool(output)}))
        for _, metadata in output:
            metadata["overlap_tokens"] = self._count("\n".join([header, separator])) if metadata.get("header_repeated") else 0
        return output

    def _split_oversized_table_row(self, header: str, separator: str, row: str, row_number: int, heading: str | None):
        headers = [cell.strip() for cell in header.strip("|").split("|")]
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        pairs = [f"{headers[index] if index < len(headers) else f'column_{index + 1}'}: {cell}" for index, cell in enumerate(cells)]
        raw = "\n".join(pairs)
        pieces = self._split_raw(raw, heading, "table_row")
        return [
            (
                piece,
                {
                    "row_start": row_number,
                    "row_end": row_number,
                    "table_split": True,
                    "oversized_row_split": True,
                    "overlap_tokens": overlap,
                },
            )
            for piece, overlap in pieces
        ]

    def _create_payload(
        self,
        raw_text: str,
        doc: Mapping[str, Any],
        index: int,
        heading: str | None,
        block_type: str,
        logical_start: int,
        logical_end: int,
        overlap_count: int,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        final_text = self._render(raw_text, heading, block_type)
        document_id = str(doc.get("document_id") or doc.get("source") or doc.get("url") or "unknown")
        version = doc.get("version", 1)
        source_part = doc.get("source_part", 0)
        content_hash = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
        identity = json.dumps(
            {
                "document_id": document_id,
                "version": version,
                "source_part": source_part,
                "chunk_index": index,
                "logical_start": logical_start,
                "logical_end": logical_end,
                "content_hash": content_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        inherited = dict(doc.get("metadata") or {})
        page_number = inherited.get("page_number", inherited.get("page", doc.get("page")))
        resolved_heading = heading or inherited.get("heading") or doc.get("heading")
        section = resolved_heading or inherited.get("section") or doc.get("section") or "General Context"
        token_count = self._count(final_text)
        metadata = {
            **inherited,
            "document_id": document_id,
            "content_hash": content_hash,
            "version": version,
            "source": doc.get("source") or inherited.get("source_path"),
            "url": doc.get("url"),
            "page": page_number,
            "page_number": page_number,
            "section": section,
            "heading": resolved_heading,
            "block_type": block_type,
            "chunk_id": chunk_id,
            "chunk_index": index,
            "logical_start": logical_start,
            "logical_end": logical_end,
            "token_count": token_count,
            "token_estimate": token_count,
            "token_limit": self.effective_max_tokens,
            "overlap_tokens": overlap_count,
            "confidence": doc.get("confidence", inherited.get("confidence")),
            "method": doc.get("method", inherited.get("parser_method")),
            "source_part": source_part,
            **dict(extra_metadata or {}),
        }
        if doc.get("indexed_at") is not None:
            metadata["indexed_at"] = doc.get("indexed_at")
        return {
            "_id": chunk_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": final_text,
            "metadata": metadata,
        }


__all__ = [
    "CallableTokenCounter",
    "ModelTokenCounter",
    "RegexTokenCounter",
    "TokenCounter",
    "UltimateChunker",
]
