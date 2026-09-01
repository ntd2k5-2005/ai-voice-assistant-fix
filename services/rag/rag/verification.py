"""Static isolation checks for the deployable :mod:`rag` package."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORBIDDEN_IMPORT_ROOTS = frozenset({"ragflow", "ragflow_main", "deepdoc"})
INTERNAL_IMPORT_ROOTS = frozenset(
    {
        "Embeddings",
        "benchmark",
        "benchmark_cli",
        "config",
        "contracts",
        "errors",
        "ingestion",
        "lifecycle",
        "metrics",
        "model_identity",
        "observability",
        "pipeline",
        "processing",
        "retrieval",
        "service",
        "storage",
        "streamlit_app",
        "verification",
    }
)


@dataclass(frozen=True)
class DependencyViolation:
    path: Path
    line: int
    message: str


def runtime_python_files(package_root: str | Path | None = None) -> Iterable[Path]:
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    excluded = {".venv", "__pycache__", "tests"}
    for path in sorted(root.rglob("*.py")):
        if any(part in excluded for part in path.parts):
            continue
        yield path


def scan_runtime(package_root: str | Path | None = None) -> list[DependencyViolation]:
    violations: list[DependencyViolation] = []
    for path in runtime_python_files(package_root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            violations.append(DependencyViolation(path, getattr(exc, "lineno", 0) or 0, f"unparseable runtime source: {exc}"))
            continue
        for node in ast.walk(tree):
            imported: list[str] = []
            absolute_import = False
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
                absolute_import = True
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
                absolute_import = node.level == 0
            for module in imported:
                root_module = module.split(".", 1)[0]
                if root_module in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(DependencyViolation(path, node.lineno, f"forbidden reference import: {module}"))
                if absolute_import and root_module in INTERNAL_IMPORT_ROOTS:
                    violations.append(
                        DependencyViolation(
                            path,
                            node.lineno,
                            f"internal package import must be relative: {module}",
                        )
                    )
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attribute = node.func
            if (
                attribute.attr in {"insert", "append"}
                and isinstance(attribute.value, ast.Attribute)
                and isinstance(attribute.value.value, ast.Name)
                and attribute.value.value.id == "sys"
                and attribute.value.attr == "path"
            ):
                violations.append(DependencyViolation(path, node.lineno, "runtime must not mutate sys.path"))
            if (
                attribute.attr == "basicConfig"
                and isinstance(attribute.value, ast.Name)
                and attribute.value.id == "logging"
            ):
                violations.append(DependencyViolation(path, node.lineno, "library must not configure global logging"))
    return violations


def assert_runtime_isolated(package_root: str | Path | None = None) -> None:
    violations = scan_runtime(package_root)
    if violations:
        details = "\n".join(f"{item.path}:{item.line}: {item.message}" for item in violations)
        raise RuntimeError(f"RAG runtime isolation failed:\n{details}")


def main() -> int:
    violations = scan_runtime()
    if violations:
        for item in violations:
            print(f"{item.path}:{item.line}: {item.message}")
        return 1
    print("RAG runtime isolation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DependencyViolation",
    "FORBIDDEN_IMPORT_ROOTS",
    "INTERNAL_IMPORT_ROOTS",
    "assert_runtime_isolated",
    "runtime_python_files",
    "scan_runtime",
]
