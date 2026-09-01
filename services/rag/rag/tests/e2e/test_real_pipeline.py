from dataclasses import replace

import pytest

from rag import DocumentSource, RagConfig, RagModule
from rag.config import ChunkingConfig, PathsConfig, RetrievalConfig


GTE_WEIGHTS_REVISION = "8215cf04918ba6f7b6a62bb44238ce2953d8831c"
GTE_CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.slow
def test_real_public_two_flows_use_default_e5_and_required_gte_offline(tmp_path):
    """Functional proof only: no relevance metric or quality claim is produced."""

    source_path = tmp_path / "knowledge.md"
    source_path.write_text(
        """# Hệ thống tri thức

Registry CAS chỉ chuyển phiên bản hiển thị sau khi dense và sparse đã được ghi,
đọc lại và xác minh. Phiên bản cũ vẫn tìm được trước thời điểm CAS.
""",
        encoding="utf-8",
    )
    base = RagConfig()
    config = replace(
        base,
        paths=PathsConfig(
            qdrant=tmp_path / "qdrant",
            sparse=tmp_path / "sparse.db",
            registry=tmp_path / "registry.db",
            benchmark_output=tmp_path / "benchmarks",
        ),
        chunking=ChunkingConfig(
            max_tokens=64,
            reserved_tokens=8,
            overlap_tokens=8,
            min_chunk_tokens=4,
        ),
        embedding=replace(
            base.embedding,
            device="cpu",
            batch_size=4,
            local_files_only=True,
        ),
        reranker=replace(
            base.reranker,
            device="cpu",
            batch_size=2,
            local_files_only=True,
            revision=GTE_WEIGHTS_REVISION,
            code_revision=GTE_CODE_REVISION,
            dtype="auto",
        ),
        retrieval=RetrievalConfig(
            default_mode="hybrid",
            top_k=1,
            candidate_k=5,
            parallel_hybrid=False,
        ),
    )

    with RagModule(config) as module:
        runtime = module.runtime_snapshot()
        assert runtime["configured"]["embedding"]["model_name"] == (
            "intfloat/multilingual-e5-small"
        )
        assert runtime["configured"]["reranker"]["model_name"] == (
            "Alibaba-NLP/gte-multilingual-reranker-base"
        )
        assert runtime["resolved"]["embedding"]["family"] == "e5"
        assert runtime["resolved"]["embedding"]["dimension"] == 384
        assert runtime["resolved"]["reranker"]["family"] == "gte-reranker"
        assert runtime["resolved"]["reranker"]["revision"] == GTE_WEIGHTS_REVISION
        assert runtime["resolved"]["reranker"]["code_revision"] == GTE_CODE_REVISION

        # A direct two-passage sanity check proves the required checkpoint's
        # score direction before it participates in the integrated flow.
        reranker = module.pipeline.reranker
        sanity = reranker.rerank(
            "What is the capital of China?",
            [
                {
                    "chunk_id": "relevant",
                    "text": "Beijing is the capital city of China.",
                },
                {"chunk_id": "irrelevant", "text": "Bananas are yellow fruit."},
            ],
            top_k=2,
        )
        assert [item[0]["chunk_id"] for item in sanity] == [
            "relevant",
            "irrelevant",
        ]
        assert sanity[0][1] > sanity[1][1]

        created = module.knowledge.ingest(
            DocumentSource(
                source_path,
                source_type="file",
                metadata={"document_id": "real-doc", "department": "research"},
            ),
            operation_id="real-add-v1",
            include_trace=True,
        )
        assert created.action == "add"
        assert created.version == 1
        assert created.chunk_count > 0
        ingest_trace = created.diagnostics["trace"]
        assert ingest_trace["source"]["parser"] == "markdown"
        assert ingest_trace["parsed_units"]
        assert ingest_trace["normalized_units"]
        assert ingest_trace["chunks"]
        assert all(
            0 < int(chunk["token_count"])
            <= runtime["resolved"]["chunking"]["effective_max_tokens"]
            for chunk in ingest_trace["chunks"]
        )
        assert ingest_trace["embedding"]["dimension"] == 384
        assert ingest_trace["embedding"]["chunk_count"] == created.chunk_count
        assert ingest_trace["embedding"]["finite"] is True
        assert ingest_trace["sparse"]["status"] == "verified_active"
        assert ingest_trace["lifecycle"]["verification_status"] == (
            "dense_sparse_parity_verified"
        )
        assert ingest_trace["lifecycle"]["active"] is True

        inspection = module.knowledge.inspect("real-doc", include_chunks=True)
        assert len(inspection["documents"]) == 1
        assert inspection["active_chunks"]
        assert {chunk["version"] for chunk in inspection["active_chunks"]} == {1}
        assert all("vector" not in chunk for chunk in inspection["active_chunks"])

        query = "Registry CAS sau bước nào?"
        dense = module.retrieve(query, mode="dense", top_k=1, candidate_k=5)
        sparse = module.retrieve(query, mode="sparse", top_k=1, candidate_k=5)
        hybrid = module.retrieve(query, mode="hybrid", top_k=1, candidate_k=5)
        reranked = module.retrieve(
            query,
            mode="hybrid",
            top_k=1,
            candidate_k=5,
            rerank=True,
            include_trace=True,
        )
        for result in (dense, sparse, hybrid, reranked):
            assert result.candidates
            assert len(result.candidates) == 1
            assert result.candidates[0].document_id == "real-doc"
            assert result.candidates[0].document_version == 1
            assert "CAS" in result.candidates[0].text
            assert result.diagnostics["candidate_k"] == 5
        retrieval_trace = reranked.diagnostics["trace"]
        assert retrieval_trace["query"]["embedding_executed"] is True
        assert retrieval_trace["dense_candidates"]
        assert retrieval_trace["sparse_candidates"]
        assert retrieval_trace["fusion"]["executed"] is True
        assert retrieval_trace["fusion"]["candidates"]
        assert retrieval_trace["rerank"]["applied"] is True
        assert retrieval_trace["rerank"]["before"]
        assert retrieval_trace["rerank"]["after"]
        assert reranked.diagnostics["rerank_fallback"] is False

        source_path.write_text(
            """# Hệ thống tri thức

Registry CAS chuyển sang phiên bản hai sau khi cả dense và sparse xác minh xong.
Dấu kiểm chứng cập nhật là ORCHID-2026.
""",
            encoding="utf-8",
        )
        updated = module.knowledge.update(
            DocumentSource(
                source_path,
                source_type="file",
                metadata={"document_id": "real-doc", "department": "research"},
            ),
            operation_id="real-update-v2",
            include_trace=True,
        )
        assert updated.action == "update"
        assert updated.version == 2
        updated_inspection = module.knowledge.inspect(
            "real-doc", include_chunks=True
        )
        assert updated_inspection["active_chunks"]
        assert {chunk["version"] for chunk in updated_inspection["active_chunks"]} == {2}
        assert module.retrieve(
            "ORCHID-2026", mode="hybrid", top_k=1, candidate_k=5
        ).candidates[0].document_version == 2
        assert module.knowledge.recover() == []

        deleted = module.knowledge.delete(
            "real-doc", operation_id="real-delete-v2"
        )
        assert deleted["action"] == "delete"
        assert not module.retrieve(
            "ORCHID-2026", mode="hybrid", top_k=1, candidate_k=5
        ).candidates
