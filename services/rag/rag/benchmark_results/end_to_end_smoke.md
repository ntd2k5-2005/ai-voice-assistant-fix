# Retrieval benchmark report

> Generated from the adjacent machine-readable artifact. Values are not hand-authored.

- Kind: `end_to_end`
- Dataset: `built-in-vietnamese-diagnostic-v1` (8 cases)
- Seed: `42`
- Timestamp (UTC): `2026-08-16T13:46:33.303076+00:00`

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 total ms | p95 total ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 27.929 | 31.949 |
| sparse | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.872 | 1.479 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 29.584 | 31.250 |
| hybrid+rerank | 0.8750 | 1.0000 | 1.0000 | 0.9375 | 0.9539 | 114.100 | 144.775 |

See the JSON and CSV files for raw rankings and per-query timings.
