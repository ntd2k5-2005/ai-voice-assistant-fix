from rag.config import RagConfig
from rag.environment import config_from_environment


def test_api_environment_applies_explicit_model_placement_only():
    central = RagConfig()
    effective = config_from_environment(
        {
            "RAG_EMBEDDING_DEVICE": "cpu",
            "RAG_EMBEDDING_DTYPE": "float32",
            "RAG_RERANKER_DEVICE": "cuda",
            "RAG_RERANKER_DTYPE": "float16",
            "RAG_RERANKER_CODE_REVISION": "code-commit",
        },
        base=central,
    )
    assert effective.embedding.model_name == central.embedding.model_name
    assert effective.embedding.device == "cpu"
    assert effective.embedding.dtype == "float32"
    assert effective.reranker.device == "cuda"
    assert effective.reranker.dtype == "float16"
    assert effective.reranker.code_revision == "code-commit"
    assert effective.retrieval is central.retrieval


def test_api_environment_can_enforce_cache_only_runtime():
    effective = config_from_environment(
        {
            "RAG_EMBEDDING_LOCAL_FILES_ONLY": "true",
            "RAG_RERANKER_LOCAL_FILES_ONLY": "true",
        }
    )
    assert effective.embedding.local_files_only is True
    assert effective.reranker.local_files_only is True
