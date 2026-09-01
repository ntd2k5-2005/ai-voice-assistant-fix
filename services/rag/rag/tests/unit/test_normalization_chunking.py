from __future__ import annotations

import re

from rag.config import ChunkingConfig, NormalizationConfig
from rag.processing.chunking import RegexTokenCounter, UltimateChunker
from rag.processing.clean_text import clean_text, normalize_document


def _word_count(value: str) -> int:
    return len(re.findall(r"\S+", value))


def test_normalization_is_idempotent_unicode_and_structure_preserving():
    source = "\ufeffTiếng Việt\r\n\r\n- mục 1\r\n  thụt lề\r\n| A | B |\r\n|---|---|\r\n\tcode\x00\r\n\r\n\r\n"
    config = NormalizationConfig(unicode_form="NFC", collapse_blank_lines=2)
    normalized = clean_text(source, config)

    assert normalized == clean_text(normalized, config)
    assert "Tiếng Việt" in normalized
    assert "- mục 1\n  thụt lề" in normalized
    assert "| A | B |\n|---|---|" in normalized
    assert "\tcode" in normalized
    assert "\x00" not in normalized


def test_normalize_document_does_not_mutate_or_change_provenance():
    metadata = {"page_number": 3, "heading": "Mục A", "nested": {"row": 2}}
    document = {"text": "Nội dung\r\n", "metadata": metadata, "source": "a.txt"}
    result = normalize_document(document)

    assert result["text"] == "Nội dung"
    assert result["metadata"] == metadata
    assert result["metadata"] is not metadata
    assert document["text"].endswith("\r\n")


def test_token_limit_overlap_long_paragraph_unicode_and_provenance():
    chunker = UltimateChunker(
        max_tokens=12,
        overlap_tokens=2,
        reserved_tokens=0,
        token_counter=_word_count,
    )
    doc = {
        "document_id": "vi-doc",
        "version": 2,
        "source": "vi.txt",
        "page": 7,
        "text": " ".join(f"từ{i}" for i in range(45)),
        "metadata": {"tenant": "qa", "heading": "Tiếng Việt", "page_number": 7},
    }
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    assert all(_word_count(chunk["text"]) <= 12 for chunk in chunks)
    assert all(chunk["metadata"]["tenant"] == "qa" for chunk in chunks)
    assert all(chunk["metadata"]["page_number"] == 7 for chunk in chunks)
    assert any(chunk["metadata"]["overlap_tokens"] > 0 for chunk in chunks[1:])


def test_heading_transition_flushes_before_metadata_update_for_all_vietnamese_labels():
    labels = ["Chương 1 Tổng quan", "Mục 2 Chi tiết", "Phần 3 Dữ liệu", "Bài 4 Ví dụ", "Câu 5 Kết luận"]
    text = "\n\n".join(f"{label}\nNội dung dành riêng cho {index}." for index, label in enumerate(labels, 1))
    chunks = UltimateChunker(max_tokens=40, overlap_tokens=0, reserved_tokens=0, token_counter=_word_count).chunk_document(
        {"document_id": "sections", "source": "sections.txt", "text": text}
    )

    assert [chunk["metadata"]["heading"] for chunk in chunks] == labels
    for index, chunk in enumerate(chunks, 1):
        assert f"riêng cho {index}" in chunk["text"]
        assert all(f"riêng cho {other}" not in chunk["text"] for other in range(1, 6) if other != index)


def test_markdown_table_splits_only_between_rows_and_repeats_header():
    text = "\n".join(
        ["| name | role |", "| --- | --- |"]
        + [f"| person-{index} | role-{index} |" for index in range(1, 9)]
    )
    chunks = UltimateChunker(max_tokens=20, overlap_tokens=1, reserved_tokens=0, token_counter=_word_count).chunk_document(
        {
            "document_id": "table",
            "source": "table.md",
            "text": text,
            "metadata": {"block_type": "table", "table_id": "employees", "column_names": ["name", "role"]},
        }
    )

    assert len(chunks) > 1
    assert all("| name | role |" in chunk["text"] for chunk in chunks)
    assert all(chunk["metadata"]["table_id"] == "employees" for chunk in chunks)
    assert all(chunk["metadata"]["row_start"] <= chunk["metadata"]["row_end"] for chunk in chunks)
    assert all(_word_count(chunk["text"]) <= 20 for chunk in chunks)


def test_huge_single_block_and_oversized_table_cell_still_respect_limit():
    chunker = UltimateChunker(max_tokens=9, overlap_tokens=2, reserved_tokens=0)
    chunks = chunker.chunk_document({"document_id": "huge", "source": "huge.txt", "text": "x" * 500})
    assert len(chunks) > 1
    assert all(RegexTokenCounter().count(chunk["text"]) <= 9 for chunk in chunks)

    table = "| key | value |\n| --- | --- |\n| one | " + ("z" * 300) + " |"
    table_chunks = chunker.chunk_document(
        {"document_id": "huge-table", "source": "table.md", "text": table, "metadata": {"block_type": "table"}}
    )
    assert all(chunk["metadata"]["oversized_row_split"] for chunk in table_chunks)
    assert all(RegexTokenCounter().count(chunk["text"]) <= 9 for chunk in table_chunks)


def test_chunk_ids_and_complete_payload_are_deterministic_but_version_sensitive():
    chunker = UltimateChunker(max_tokens=10, overlap_tokens=1, reserved_tokens=0, token_counter=_word_count)
    document = {"document_id": "stable", "version": 3, "source": "a.txt", "text": "one two three four five six seven eight nine ten eleven"}
    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)
    changed_version = chunker.chunk_document({**document, "version": 4})

    assert first == second
    assert [chunk["chunk_id"] for chunk in first] != [chunk["chunk_id"] for chunk in changed_version]


def test_model_limit_resolution_and_actual_tokenizer_counter():
    class CharacterTokenizer:
        @staticmethod
        def encode(value, add_special_tokens=False):
            return list(value)

    config = ChunkingConfig(
        max_tokens=None,
        fallback_max_tokens=50,
        overlap_tokens=2,
        reserved_tokens=3,
        min_chunk_tokens=1,
    )
    chunker = UltimateChunker(config=config, model_max_tokens=20, tokenizer=CharacterTokenizer())
    chunks = chunker.chunk_document({"document_id": "model", "source": "m.txt", "text": "abcdefghij " * 8})

    assert chunker.effective_max_tokens == 17
    assert all(len(chunk["text"]) <= 17 for chunk in chunks)


def test_min_chunk_tokens_merges_compatible_adjacent_units_without_data_loss():
    def produce(minimum):
        chunker = UltimateChunker(
            config=ChunkingConfig(
                max_tokens=10,
                overlap_tokens=0,
                reserved_tokens=0,
                min_chunk_tokens=minimum,
            ),
            token_counter=_word_count,
        )
        parts = [
            {
                "document_id": "short-tail",
                "version": 1,
                "source": "short.txt",
                "source_part": 0,
                "text": "one two three four five",
                "metadata": {"source_part": 0, "heading": "Same section"},
            },
            {
                "document_id": "short-tail",
                "version": 1,
                "source": "short.txt",
                "source_part": 1,
                "text": "six",
                "metadata": {"source_part": 1, "heading": "Same section"},
            },
        ]
        raw = [chunk for part in parts for chunk in chunker.chunk_document(part)]
        return chunker.finalize_document_chunks(raw)

    without_merge = produce(1)
    with_merge = produce(4)

    assert len(without_merge) == 2
    assert len(with_merge) == 1
    assert with_merge[0]["text"] == "[Same section]\none two three four five\n\nsix"
    assert with_merge[0]["metadata"]["token_count"] <= 10
    assert with_merge[0]["metadata"]["merged_short_chunks"] == 2
    assert with_merge[0]["metadata"]["source_parts"] == [0, 1]
    assert with_merge[0]["metadata"]["below_min_tokens"] is False


def test_min_chunk_tokens_retains_and_marks_short_hard_boundary():
    chunker = UltimateChunker(
        config=ChunkingConfig(
            max_tokens=12,
            overlap_tokens=0,
            reserved_tokens=0,
            min_chunk_tokens=4,
        ),
        token_counter=_word_count,
    )
    first = chunker.chunk_document(
        {
            "document_id": "boundaries",
            "source": "pages.txt",
            "source_part": 0,
            "page": 1,
            "text": "enough words remain on page one",
            "metadata": {"source_part": 0, "page_number": 1, "heading": "A"},
        }
    )
    second = chunker.chunk_document(
        {
            "document_id": "boundaries",
            "source": "pages.txt",
            "source_part": 1,
            "page": 2,
            "text": "tiny",
            "metadata": {"source_part": 1, "page_number": 2, "heading": "B"},
        }
    )

    result = chunker.finalize_document_chunks([*first, *second])
    assert len(result) == 2
    assert result[1]["text"].endswith("tiny")
    assert result[1]["metadata"]["below_min_tokens"] is True
    assert result[1]["metadata"]["min_chunk_tokens"] == 4


def test_batch_chunking_turns_heading_only_units_into_hierarchical_context():
    chunker = UltimateChunker(
        config=ChunkingConfig(
            max_tokens=80,
            overlap_tokens=12,
            reserved_tokens=0,
            min_chunk_tokens=8,
        ),
        token_counter=_word_count,
    )
    units = [
        {
            "document_id": "solutions",
            "version": 1,
            "source": "report.docx",
            "source_part": 0,
            "text": "Giải pháp, biện pháp khắc phục",
            "metadata": {
                "source_part": 0,
                "block_type": "section",
                "heading": "Giải pháp, biện pháp khắc phục",
                "section": "Giải pháp, biện pháp khắc phục",
                "page_number": 6,
            },
        },
        {
            "document_id": "solutions",
            "version": 1,
            "source": "report.docx",
            "source_part": 1,
            "text": (
                "Đối với cá nhân\n"
                "Để khắc phục tình trạng sợ sai, mỗi cá nhân cần thay đổi nhận thức.\n"
                "Bên cạnh đó, cần rèn luyện kỹ năng giao tiếp và phản biện."
            ),
            "metadata": {
                "source_part": 1,
                "block_type": "section",
                "heading": "Đối với cá nhân",
                "section": "Đối với cá nhân",
                "page_number": 7,
            },
        },
        {
            "document_id": "solutions",
            "version": 1,
            "source": "report.docx",
            "source_part": 2,
            "text": (
                "Đối với gia đình\n"
                "Gia đình cần tạo không gian đối thoại cởi mở và tôn trọng người trẻ."
            ),
            "metadata": {
                "source_part": 2,
                "block_type": "section",
                "heading": "Đối với gia đình",
                "section": "Đối với gia đình",
                "page_number": 8,
            },
        },
    ]

    chunks = chunker.chunk_documents(units)
    texts = [chunk["text"] for chunk in chunks]
    assert "Giải pháp, biện pháp khắc phục" not in texts
    assert all(chunk["metadata"]["below_min_tokens"] is False for chunk in chunks)
    assert chunks[0]["metadata"]["heading_path"] == [
        "Giải pháp, biện pháp khắc phục",
        "Đối với cá nhân",
    ]
    assert chunks[1]["metadata"]["heading_path"] == [
        "Giải pháp, biện pháp khắc phục",
        "Đối với gia đình",
    ]
    assert chunks[0]["text"].startswith(
        "[Giải pháp, biện pháp khắc phục > Đối với cá nhân]\n"
    )


def test_batch_chunking_recovers_internal_subsections_and_never_glues_boundaries():
    chunker = UltimateChunker(
        config=ChunkingConfig(
            max_tokens=55,
            overlap_tokens=8,
            reserved_tokens=0,
            min_chunk_tokens=8,
        ),
        token_counter=_word_count,
    )
    unit = {
        "document_id": "causes",
        "version": 1,
        "source": "report.docx",
        "source_part": 12,
        "text": (
            "Nguyên nhân\n"
            "Từ bản thân\n"
            "Một trong những nguyên nhân quan trọng là sự thiếu tự tin và nỗi sợ bị đánh giá.\n"
            "Từ nhà trường\n"
            "Môi trường học tập cũng có ảnh hưởng lớn đến việc hình thành tâm lý trên.\n"
            "Từ xã hội và môi trường làm việc\n"
            "Bên cạnh các yếu tố trên, môi trường thiếu an toàn tâm lý làm người trẻ ngại phản biện."
        ),
        "metadata": {
            "source_part": 12,
            "block_type": "section",
            "heading": "Nguyên nhân",
            "section": "Nguyên nhân",
            "page_number": 4,
        },
    }

    chunks = chunker.chunk_documents([unit])
    paths = [chunk["metadata"].get("heading_path") for chunk in chunks]
    assert ["Nguyên nhân", "Từ bản thân"] in paths
    assert ["Nguyên nhân", "Từ nhà trường"] in paths
    assert ["Nguyên nhân", "Từ xã hội và môi trường làm việc"] in paths
    combined = "\n".join(chunk["text"] for chunk in chunks)
    assert "Từ nhà trườngMôi trường" not in combined
    assert "làm việcBên cạnh" not in combined


def test_overlap_prefers_complete_semantic_units_over_sentence_fragments():
    chunker = UltimateChunker(
        max_tokens=18,
        overlap_tokens=4,
        reserved_tokens=0,
        token_counter=_word_count,
    )
    chunks = chunker.chunk_document(
        {
            "document_id": "sentences",
            "source": "sentences.txt",
            "text": (
                "Câu thứ nhất có đủ nhiều từ để tạo thành một câu hoàn chỉnh. "
                "Câu thứ hai cũng có đủ nhiều từ để kiểm tra ranh giới. "
                "Câu thứ ba tiếp tục nội dung của tài liệu."
            ),
        }
    )
    # When a complete sentence cannot fit the small overlap budget, no fragment
    # is injected at the front of the following chunk.
    assert all(not chunk["text"].startswith(("nhiều từ", "đủ nhiều", "thành một")) for chunk in chunks[1:])


def test_batch_chunking_preserves_explicit_parser_heading_paths_across_siblings():
    chunker = UltimateChunker(
        config=ChunkingConfig(
            max_tokens=80, overlap_tokens=0, reserved_tokens=0, min_chunk_tokens=4
        ),
        token_counter=_word_count,
    )
    units = [
        {
            "document_id": "structured",
            "source": "guide.md",
            "source_part": 0,
            "text": "Install dependencies and initialize the service correctly.",
            "metadata": {
                "source_part": 0,
                "heading": "Install",
                "heading_path": ["Setup", "Install"],
                "heading_level": 2,
            },
        },
        {
            "document_id": "structured",
            "source": "guide.md",
            "source_part": 1,
            "text": "Configure the runtime before starting the service.",
            "metadata": {
                "source_part": 1,
                "heading": "Configure",
                "heading_path": ["Setup", "Configure"],
                "heading_level": 2,
            },
        },
    ]

    chunks = chunker.chunk_documents(units)
    assert [chunk["metadata"]["heading_path"] for chunk in chunks] == [
        ["Setup", "Install"],
        ["Setup", "Configure"],
    ]
    assert chunks[0]["text"].startswith("[Setup > Install]\n")
    assert chunks[1]["text"].startswith("[Setup > Configure]\n")
