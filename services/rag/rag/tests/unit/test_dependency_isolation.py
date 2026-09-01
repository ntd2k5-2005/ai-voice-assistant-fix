from pathlib import Path
import subprocess
import sys

from rag.verification import scan_runtime


def test_runtime_has_no_ragflow_dependency_or_path_mutation():
    root = Path(__file__).resolve().parents[2]
    assert scan_runtime(root) == []


def test_scanner_rejects_top_level_internal_imports_but_accepts_relative(tmp_path):
    (tmp_path / "bad.py").write_text(
        "from contracts import RetrievalResult\nimport storage.identity\n",
        encoding="utf-8",
    )
    (tmp_path / "good.py").write_text(
        "from .contracts import RetrievalResult\nfrom .storage import identity\n",
        encoding="utf-8",
    )
    violations = scan_runtime(tmp_path)
    assert len(violations) == 2
    assert all("must be relative" in violation.message for violation in violations)
    assert {violation.line for violation in violations} == {1, 2}


def test_sparse_module_does_not_import_dense_model_or_ocr_dependencies():
    script = r'''
import importlib.abc
import sys

blocked = {
    "qdrant_client",
    "sentence_transformers",
    "transformers",
    "pytesseract",
    "pdfplumber",
}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"unexpected dependency import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import rag.retrieval.bm25
assert not blocked.intersection(sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_plain_text_parser_does_not_import_model_or_dense_store_dependencies():
    script = r'''
import importlib.abc
import sys

blocked = {"qdrant_client", "sentence_transformers", "transformers"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"unexpected dependency import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import rag.ingestion.parsers.text
assert not blocked.intersection(sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
