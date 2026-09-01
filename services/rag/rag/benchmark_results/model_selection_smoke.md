# Retrieval benchmark report

> Generated from the adjacent machine-readable artifact. Values are not hand-authored.

- Kind: `model_selection`
- Dataset: `built-in-vietnamese-diagnostic-v1` (8 cases)
- Seed: `42`
- Timestamp (UTC): `2026-08-16T13:46:28.101217+00:00`

| Candidate | Component | Recall@1 | Recall@5 | MRR | nDCG@10 | Mean latency ms | Dimension | Max length | Device |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| intfloat/multilingual-e5-small | embedding | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.953 | 384 | 512 | cpu |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | embedding | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6.537 | 384 | 128 | cpu |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | reranker | 0.8750 | 1.0000 | 0.9375 | 0.9539 | 102.376 | n/a | 512 | cpu |

Model-selection values use fixed document/query or candidate pools; they are not end-to-end service timings.
See the JSON and CSV files for raw rankings and per-query evidence.
