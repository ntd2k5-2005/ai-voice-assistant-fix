# Benchmark report

> **Historical pre-hardening artifact.** This report is retained for
> reproducibility only. It was not rerun during final module completion, its
> MS-MARCO reranker is not the current required Alibaba GTE runtime default,
> and none of its ranking metrics are used as completion evidence.

> Generated from the two linked JSON artifacts by `rag.benchmark_cli.write_combined_report`; no metric in this file is hand-entered.

## Scope and method

- Dataset: `built-in-vietnamese-diagnostic-v1`; 8 documents, 8 Vietnamese queries.
- Seed: `42`.
- Platform: `Windows-11-10.0.26200-SP0`.
- Model-selection timestamp: `2026-08-16T13:46:28.101217+00:00`.
- End-to-end timestamp: `2026-08-16T13:46:33.303076+00:00`.
- The corpus is an authored diagnostic smoke fixture. It is too small to support a general model-quality claim or production SLO.
- Query timings are warmed, single-process, local-machine measurements; storage uses temporary local Qdrant plus SQLite FTS5.

## A. Model-selection smoke

| Candidate | Component | Recall@1 | Recall@5 | MRR | nDCG@10 | Mean query/rerank ms | Dimension | Max length | Device |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | reranker | 0.8750 | 1.0000 | 0.9375 | 0.9539 | 102.376 | n/a | 512 | cpu |
| intfloat/multilingual-e5-small | embedding | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.953 | 384 | 512 | cpu |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | embedding | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6.537 | 384 | 128 | cpu |

All embedding candidates ranked every relevant passage first in this smoke fixture. This tiny result is not proof of equivalence. `intfloat/multilingual-e5-small` had the lowest measured mean query-embedding latency in this run. The configured default remains multilingual E5 because its resolved profile has explicit query/document semantics and a 512-token limit; production selection still requires representative labels.

`cross-encoder/ms-marco-MiniLM-L-6-v2` produced Recall@1=0.8750 and Recall@5=1.0000 on its fixed candidate pools. Therefore reranking remains explicit/optional and its fallback/diagnostics are surfaced; this run does not by itself justify enabling that reranker by default.

## B. End-to-end smoke

| Mode | Recall@1 | Recall@5 | MRR | nDCG@10 | p50 total ms | p95 total ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 27.929 | 31.949 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 29.584 | 31.250 |
| hybrid+rerank | 0.8750 | 1.0000 | 0.9375 | 0.9539 | 114.100 | 144.775 |
| sparse | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.872 | 1.479 |

Modes with Recall@1=1.0 on this deliberately separable corpus: dense, hybrid, sparse. Hybrid+rerank produced Recall@1=0.8750. Per-stage latency, raw rankings, scores, configuration snapshot, package versions, and per-query metrics are retained in the raw artifacts.

## Evidence

- [Model-selection JSON](benchmark_results/model_selection_smoke.json) — SHA-256 `481fd765f87742b904b7ac5e81ce3d329ceec53763e4d4d98d0a8e8a20273d23`
- [End-to-end JSON](benchmark_results/end_to_end_smoke.json) — SHA-256 `0c987f9653ff35917ebaac7227d80e1ca66690aa86c05f9a60a59cb29843dcd2`
- [Model-selection CSV](benchmark_results/model_selection_smoke.csv)
- [End-to-end CSV](benchmark_results/end_to_end_smoke.csv)

Reproduce locally (cached-only by default):

```powershell
rag\.venv\Scripts\python.exe -m rag.benchmark_cli --suite all --output-dir rag\benchmark_results
```

Pass `--allow-download` only when network model downloads are intentionally authorized.
