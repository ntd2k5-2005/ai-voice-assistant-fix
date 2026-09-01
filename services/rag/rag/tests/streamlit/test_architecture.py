from __future__ import annotations

import ast
from pathlib import Path


UI_ROOT = Path(__file__).resolve().parents[2] / "streamlit_app"


def _production_sources():
    return [path for path in UI_ROOT.rglob("*.py") if "__pycache__" not in path.parts]


def test_ui_contains_no_hidden_model_or_retrieval_defaults():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in _production_sources())
    assert "multilingual-e5-small" not in combined
    assert "gte-multilingual-reranker-base" not in combined
    assert "ms-marco-MiniLM" not in combined
    assert "candidate_k=20" not in combined
    assert "top_k=5" not in combined


def test_ui_does_not_import_or_construct_storage_or_pipeline_implementations():
    forbidden_modules = {
        "rag.pipeline",
        "rag.storage",
        "rag.lifecycle",
        "rag.retrieval",
        "qdrant_client",
    }
    forbidden_names = {
        "QdrantStore",
        "SQLiteBM25Retriever",
        "DocumentRegistry",
        "RetrievalPipeline",
        "RetrievalEngine",
    }
    for path in _production_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name == module or alias.name.startswith(f"{module}.")
                    for alias in node.names
                    for module in forbidden_modules
                ), f"{path} imports a private runtime implementation"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for forbidden in forbidden_modules
                ), f"{path} imports a private runtime implementation"
                assert not ({alias.name for alias in node.names} & forbidden_names), (
                    f"{path} imports a private runtime class"
                )
            elif isinstance(node, ast.Attribute):
                assert node.attr not in {"pipeline", "store", "registry", "sparse_retriever"}, (
                    f"{path} reaches through the public facade via .{node.attr}"
                )


def test_all_streamlit_sources_are_valid_utf8_without_mojibake_markers():
    markers = ("Ã", "Â", "â†", "áº", "Ä‘")
    for path in _production_sources():
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in markers), f"mojibake in {path}"
