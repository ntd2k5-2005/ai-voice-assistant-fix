"""Command-line model-selection and end-to-end benchmark suites.

The built-in Vietnamese corpus is intentionally small and diagnostic. It is
useful for contract/regression evidence, not a claim of production quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .benchmark import (
    BenchmarkCase,
    benchmark_modes,
    evaluate_embedding_model,
    evaluate_reranker_model,
    make_artifact,
    write_artifact,
)
from .config import (
    EmbeddingConfig,
    PathsConfig,
    RagConfig,
    RerankerConfig,
    RetrievalConfig,
)
from .contracts import DocumentSource
from .service import RagModule


DOCUMENTS = {
    "vi_capital": (
        "Hà Nội là thủ đô của Việt Nam, nằm bên bờ sông Hồng và là trung tâm "
        "chính trị, văn hóa quan trọng của cả nước."
    ),
    "da_nang": (
        "Đà Nẵng là thành phố ven biển miền Trung, nổi tiếng với cầu Rồng, "
        "bãi biển Mỹ Khê và bán đảo Sơn Trà."
    ),
    "mekong": (
        "Đồng bằng sông Cửu Long là vùng sản xuất lúa gạo và thủy sản lớn ở "
        "phía Nam Việt Nam, chịu ảnh hưởng của biến đổi khí hậu."
    ),
    "photosynthesis": (
        "Quang hợp là quá trình thực vật dùng năng lượng ánh sáng để chuyển "
        "carbon dioxide và nước thành chất hữu cơ, đồng thời giải phóng oxy."
    ),
    "python": (
        "Python là ngôn ngữ lập trình bậc cao, cú pháp rõ ràng, thường được "
        "dùng trong khoa học dữ liệu, tự động hóa và phát triển dịch vụ web."
    ),
    "qdrant": (
        "Qdrant là cơ sở dữ liệu vector hỗ trợ tìm kiếm tương đồng và lọc "
        "metadata, thường được dùng làm dense store trong hệ thống truy hồi."
    ),
    "diabetes": (
        "Đái tháo đường là bệnh chuyển hóa mạn tính đặc trưng bởi đường huyết "
        "cao; chẩn đoán và điều trị cần được thực hiện bởi nhân viên y tế."
    ),
    "water_cycle": (
        "Vòng tuần hoàn nước gồm bay hơi, ngưng tụ, mưa và dòng chảy, liên tục "
        "trao đổi nước giữa đại dương, khí quyển và đất liền."
    ),
}

CASES = (
    BenchmarkCase("Thủ đô Việt Nam là thành phố nào?", frozenset({"vi_capital"}), case_id="q01"),
    BenchmarkCase("Thành phố nào có bãi biển Mỹ Khê?", frozenset({"da_nang"}), case_id="q02"),
    BenchmarkCase("Vùng trồng lúa lớn ở miền Nam", frozenset({"mekong"}), case_id="q03"),
    BenchmarkCase("Thực vật tạo chất hữu cơ nhờ ánh sáng như thế nào?", frozenset({"photosynthesis"}), case_id="q04"),
    BenchmarkCase("Ngôn ngữ lập trình dùng cho khoa học dữ liệu", frozenset({"python"}), case_id="q05"),
    BenchmarkCase("Cơ sở dữ liệu vector có lọc metadata", frozenset({"qdrant"}), case_id="q06"),
    BenchmarkCase("Bệnh mạn tính có đường huyết cao", frozenset({"diabetes"}), case_id="q07"),
    BenchmarkCase("Các giai đoạn bay hơi ngưng tụ và mưa", frozenset({"water_cycle"}), case_id="q08"),
)


def _reranker_cases() -> list[dict[str, Any]]:
    candidates = [
        {"id": document_id, "document_id": document_id, "text": text}
        for document_id, text in DOCUMENTS.items()
    ]
    return [
        {
            "case_id": case.case_id,
            "query": case.query,
            "relevant": sorted(case.relevant),
            "candidates": candidates,
        }
        for case in CASES
    ]


def _accelerator_memory() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"device": "unknown", "peak_vram_mb": None}
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "peak_vram_mb": None,
            "note": "VRAM is not applicable to this CPU run",
        }
    return {
        "device": torch.cuda.get_device_name(0),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "peak_reserved_vram_mb": float(torch.cuda.max_memory_reserved() / 1024**2),
    }


def _reset_accelerator_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        return


def run_model_selection(
    output_prefix: str | Path,
    *,
    embedding_models: list[str],
    reranker_model: str,
    local_files_only: bool,
    seed: int,
) -> dict[str, Path]:
    from .Embeddings.embeddings import EmbeddingEngine
    from .retrieval.reranking import Reranker

    reports: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for model_name in embedding_models:
        engine = EmbeddingEngine(
            model_name,
            device="auto",
            batch_size="auto",
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        try:
            _reset_accelerator_peak()
            report = evaluate_embedding_model(engine, DOCUMENTS, CASES, top_k=5)
            report["component"] = "embedding"
            report["resolved_config"] = engine.resolved_config
            report["accelerator_memory"] = _accelerator_memory()
            reports[model_name] = report
            identities[model_name] = engine.resolved_config
        finally:
            engine.close()

    reranker = Reranker(
        reranker_model,
        device="auto",
        batch_size="auto",
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    try:
        _reset_accelerator_peak()
        report = evaluate_reranker_model(reranker, _reranker_cases(), top_k=5)
        report["component"] = "reranker"
        report["resolved_config"] = reranker.resolved_config
        report["accelerator_memory"] = _accelerator_memory()
        reports[reranker_model] = report
        identities[reranker_model] = reranker.resolved_config
    finally:
        reranker.close()

    artifact = make_artifact(
        benchmark_kind="model_selection",
        dataset={
            "name": "built-in-vietnamese-diagnostic-v1",
            "language": "vi",
            "documents": len(DOCUMENTS),
            "cases": len(CASES),
            "provenance": "authored diagnostic fixtures bundled with rag-service",
            "limitations": "small smoke corpus; not a production model ranking",
        },
        modes=reports,
        config_snapshot={
            "local_files_only": local_files_only,
            "top_k": 5,
            "embedding_candidates": embedding_models,
            "reranker_candidate": reranker_model,
        },
        seed=seed,
        model_identity=identities,
    )
    return write_artifact(artifact, output_prefix)


def run_end_to_end(
    output_prefix: str | Path,
    *,
    embedding_model: str,
    reranker_model: str,
    local_files_only: bool,
    seed: int,
) -> dict[str, Path]:
    with TemporaryDirectory(prefix="rag-benchmark-") as directory:
        root = Path(directory)
        config = RagConfig(
            paths=PathsConfig(
                qdrant=root / "qdrant",
                sparse=root / "sparse.db",
                registry=root / "registry.db",
                benchmark_output=Path(output_prefix).parent,
            ),
            embedding=EmbeddingConfig(
                model_name=embedding_model,
                device="auto",
                batch_size=None,
                local_files_only=local_files_only,
                trust_remote_code=False,
            ),
            reranker=RerankerConfig(
                model_name=reranker_model,
                device="auto",
                batch_size=None,
                local_files_only=local_files_only,
                trust_remote_code=False,
                failure_policy="raise",
            ),
            retrieval=RetrievalConfig(
                top_k=5,
                candidate_k=len(DOCUMENTS),
                parallel_hybrid=False,
            ),
        )
        module = RagModule(config)
        try:
            for document_id, text in DOCUMENTS.items():
                module.knowledge.ingest(
                    DocumentSource(
                        text,
                        source_type="text",
                        metadata={"document_id": document_id, "language": "vi"},
                    )
                )

            # Explicit warm-up keeps first-load/tokenizer effects out of query
            # timings while remaining visible in the artifact configuration.
            module.retrieve(CASES[0].query, mode="dense", top_k=1, candidate_k=5)
            module.retrieve(CASES[0].query, mode="sparse", top_k=1, candidate_k=5)
            module.retrieve(CASES[0].query, mode="hybrid", top_k=1, candidate_k=5)
            module.retrieve(
                CASES[0].query,
                mode="hybrid",
                top_k=1,
                candidate_k=5,
                rerank=True,
            )

            retrievers = {
                "dense": lambda query, **kwargs: module.retrieve(
                    query, mode="dense", rerank=False, **kwargs
                ),
                "sparse": lambda query, **kwargs: module.retrieve(
                    query, mode="sparse", rerank=False, **kwargs
                ),
                "hybrid": lambda query, **kwargs: module.retrieve(
                    query, mode="hybrid", rerank=False, **kwargs
                ),
                "hybrid+rerank": lambda query, **kwargs: module.retrieve(
                    query, mode="hybrid", rerank=True, **kwargs
                ),
            }
            reports = benchmark_modes(
                retrievers,
                CASES,
                top_k=5,
                candidate_k=len(DOCUMENTS),
                identity_field="document_id",
            )
            embedding_identity = getattr(module.pipeline.embedder, "resolved_config", {})
            reranker_identity = getattr(module.pipeline.reranker, "resolved_config", {})
            snapshot = config.snapshot()
            snapshot["paths"] = {
                key: "<temporary benchmark workspace>"
                if key in {"qdrant", "sparse", "registry"}
                else value
                for key, value in snapshot["paths"].items()
            }
            artifact = make_artifact(
                benchmark_kind="end_to_end",
                dataset={
                    "name": "built-in-vietnamese-diagnostic-v1",
                    "language": "vi",
                    "documents": len(DOCUMENTS),
                    "cases": len(CASES),
                    "identity_field": "document_id",
                    "provenance": "authored diagnostic fixtures bundled with rag-service",
                    "limitations": "small warmed smoke run; latency is local-machine specific",
                },
                modes=reports,
                config_snapshot={**snapshot, "warmup_queries_per_mode": 1},
                seed=seed,
                model_identity={
                    "embedding": embedding_identity,
                    "reranker": reranker_identity,
                    "accelerator_memory": _accelerator_memory(),
                },
            )
            return write_artifact(artifact, output_prefix)
        finally:
            module.close()


def write_combined_report(
    model_artifact_path: str | Path,
    end_to_end_artifact_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Generate the repository benchmark summary solely from raw JSON."""

    model_path = Path(model_artifact_path)
    e2e_path = Path(end_to_end_artifact_path)
    target = Path(output_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    e2e = json.loads(e2e_path.read_text(encoding="utf-8"))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(target.parent.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    lines = [
        "# Benchmark report",
        "",
        "> Generated from the two linked JSON artifacts by `rag.benchmark_cli.write_combined_report`; no metric in this file is hand-entered.",
        "",
        "## Scope and method",
        "",
        f"- Dataset: `{e2e['dataset']['name']}`; {e2e['dataset']['documents']} documents, {e2e['dataset']['cases']} Vietnamese queries.",
        f"- Seed: `{e2e['environment']['seed']}`.",
        f"- Platform: `{e2e['environment']['platform']}`.",
        f"- Model-selection timestamp: `{model['environment']['timestamp_utc']}`.",
        f"- End-to-end timestamp: `{e2e['environment']['timestamp_utc']}`.",
        "- The corpus is an authored diagnostic smoke fixture. It is too small to support a general model-quality claim or production SLO.",
        "- Query timings are warmed, single-process, local-machine measurements; storage uses temporary local Qdrant plus SQLite FTS5.",
        "",
        "## A. Model-selection smoke",
        "",
        "| Candidate | Component | Recall@1 | Recall@5 | MRR | nDCG@10 | Mean query/rerank ms | Dimension | Max length | Device |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, report in model["modes"].items():
        quality = report["quality"]
        resolved = report.get("resolved_config", {})
        component = report.get("component", "unknown")
        latency = report["latency_ms"]
        mean = (
            latency.get("query_embedding_mean", 0.0)
            if component == "embedding"
            else latency.get("mean", 0.0)
        )
        lines.append(
            f"| {name} | {component} | {quality['recall@1']:.4f} | "
            f"{quality['recall@5']:.4f} | {quality['mrr']:.4f} | "
            f"{quality['ndcg@10']:.4f} | {mean:.3f} | "
            f"{report.get('dimension', resolved.get('dimension', 'n/a'))} | "
            f"{resolved.get('max_length', 'n/a')} | {resolved.get('device', 'n/a')} |"
        )
    embedding_reports = [
        (name, report)
        for name, report in model["modes"].items()
        if report.get("component") == "embedding"
    ]
    fastest_embedding = min(
        embedding_reports,
        key=lambda item: float(
            item[1].get("latency_ms", {}).get("query_embedding_mean", float("inf"))
        ),
    )[0]
    embeddings_all_first = all(
        float(report["quality"].get("recall@1", 0.0)) == 1.0
        for _, report in embedding_reports
    )
    reranker_reports = [
        (name, report)
        for name, report in model["modes"].items()
        if report.get("component") == "reranker"
    ]
    reranker_name, reranker_report = reranker_reports[0]
    reranker_quality = reranker_report["quality"]
    lines.extend(
        [
            "",
            (
                "All embedding candidates ranked every relevant passage first in this smoke fixture. "
                if embeddings_all_first
                else "Embedding candidates did not all rank every relevant passage first in this smoke fixture. "
            )
            + "This tiny result is not proof of equivalence. "
            + f"`{fastest_embedding}` had the lowest measured mean query-embedding latency in this run. "
            + "The configured default remains multilingual E5 because its resolved profile has explicit query/document semantics and a 512-token limit; production selection still requires representative labels.",
            "",
            f"`{reranker_name}` produced Recall@1={float(reranker_quality.get('recall@1', 0.0)):.4f} and Recall@5={float(reranker_quality.get('recall@5', 0.0)):.4f} on its fixed candidate pools. Therefore reranking remains explicit/optional and its fallback/diagnostics are surfaced; this run does not by itself justify enabling that reranker by default.",
            "",
            "## B. End-to-end smoke",
            "",
            "| Mode | Recall@1 | Recall@5 | MRR | nDCG@10 | p50 total ms | p95 total ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, report in e2e["modes"].items():
        quality = report["quality"]
        total = report["latency_ms"]["total_ms"]
        lines.append(
            f"| {mode} | {quality['recall@1']:.4f} | {quality['recall@5']:.4f} | "
            f"{quality['mrr']:.4f} | {quality['ndcg@10']:.4f} | "
            f"{total['p50']:.3f} | {total['p95']:.3f} |"
        )
    perfect_modes = [
        mode
        for mode, report in e2e["modes"].items()
        if float(report["quality"].get("recall@1", 0.0)) == 1.0
    ]
    reranked_mode = e2e["modes"].get("hybrid+rerank")
    lines.extend(
        [
            "",
            f"Modes with Recall@1=1.0 on this deliberately separable corpus: {', '.join(perfect_modes) or 'none'}. "
            + (
                f"Hybrid+rerank produced Recall@1={float(reranked_mode['quality'].get('recall@1', 0.0)):.4f}. "
                if reranked_mode is not None
                else ""
            )
            + "Per-stage latency, raw rankings, scores, configuration snapshot, package versions, and per-query metrics are retained in the raw artifacts.",
            "",
            "## Evidence",
            "",
            f"- [Model-selection JSON]({relative(model_path)}) — SHA-256 `{digest(model_path)}`",
            f"- [End-to-end JSON]({relative(e2e_path)}) — SHA-256 `{digest(e2e_path)}`",
            f"- [Model-selection CSV]({relative(model_path.with_suffix('.csv'))})",
            f"- [End-to-end CSV]({relative(e2e_path.with_suffix('.csv'))})",
            "",
            "Reproduce locally (cached-only by default):",
            "",
            "```powershell",
            r"rag\.venv\Scripts\python.exe -m rag.benchmark_cli --suite all --output-dir rag\benchmark_results",
            "```",
            "",
            "Pass `--allow-download` only when network model downloads are intentionally authorized.",
            "",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def build_argument_parser() -> argparse.ArgumentParser:
    central = RagConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("models", "e2e", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rag/benchmark_results"),
    )
    parser.add_argument(
        "--embedding-model",
        action="append",
        dest="embedding_models",
        help="Repeat to compare embedding candidates",
    )
    parser.add_argument(
        "--reranker-model",
        default=central.reranker.model_name,
        help="Defaults to RagConfig.reranker.model_name",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary-report",
        type=Path,
        default=Path("rag/BENCHMARK_REPORT.md"),
        help="Combined report generated when --suite=all",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    embedding_models = args.embedding_models or [RagConfig().embedding.model_name]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Path]] = {}
    if args.suite in {"models", "all"}:
        outputs["models"] = run_model_selection(
            args.output_dir / "model_selection_smoke",
            embedding_models=embedding_models,
            reranker_model=args.reranker_model,
            local_files_only=not args.allow_download,
            seed=args.seed,
        )
    if args.suite in {"e2e", "all"}:
        outputs["e2e"] = run_end_to_end(
            args.output_dir / "end_to_end_smoke",
            embedding_model=embedding_models[0],
            reranker_model=args.reranker_model,
            local_files_only=not args.allow_download,
            seed=args.seed,
        )
    if args.suite == "all":
        report = write_combined_report(
            outputs["models"]["json"],
            outputs["e2e"]["json"],
            args.summary_report,
        )
        print(f"summary={report}")
    for suite, paths in outputs.items():
        for kind, path in paths.items():
            print(f"{suite}.{kind}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASES",
    "DOCUMENTS",
    "build_argument_parser",
    "main",
    "run_end_to_end",
    "run_model_selection",
    "write_combined_report",
]
