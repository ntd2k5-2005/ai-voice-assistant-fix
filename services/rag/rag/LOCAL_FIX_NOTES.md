# Local review fixes — V2

This pass hardens the two things exposed by the real Streamlit trace before a
future FastAPI wrapper: semantic chunk construction and a stable public score
contract. No model-quality benchmark was run.

## Chunking / Flow 1

- Production ingestion now chunks the ordered normalized parser-unit sequence
  through `UltimateChunker.chunk_documents(...)` instead of independently
  chunking each parser unit.
- Short heading-only units are structural context and are not emitted as
  searchable knowledge when their section has following content.
- Parser-provided multi-level `heading_path` is authoritative when available.
  For reports whose formatting loses heading levels, conservative fallback
  relations recover common parent/child sections such as:
  - `Thực trạng > Nguyên nhân > Từ nhà trường`
  - `Giải pháp, biện pháp khắc phục > Đối với cá nhân`
- The structural path is kept in metadata and compactly prepended to chunk text
  so embedding/reranking receives the same context that provenance exposes.
- Internal report subsections are split conservatively rather than leaving a
  whole multi-topic section as one opaque block.
- Overlap now prefers paragraph, line, then sentence boundaries. If a complete
  semantic unit cannot fit the overlap budget, structured prose uses no overlap
  instead of starting the next chunk in the middle of a sentence.
- Adjacent text uses explicit separators. The old failures
  `Từ nhà trườngMôi trường...`, `...làm việcBên cạnh...`, and
  `Tiểu kếtCó thể...` are covered by regression tests.
- Punctuation-only labels are no longer inferred as DOCX headings.
- Ingestion/Streamlit traces now expose `heading_path`, `heading_level`,
  `parent_section`, and `section_depth` when available.

## Retrieval score contract / FastAPI readiness

Normal public candidate scores are now finite relevance scores in `[0, 1]`.
They are not calibrated probabilities.

- dense: raw cosine -> `(cosine + 1) / 2`, clamped to `[0, 1]`;
- sparse: FTS5/BM25 raw score -> query-local min-max relevance;
- hybrid default: RRF, normalized by its theoretical per-query maximum;
- weighted hybrid remains available and blends normalized branch scores;
- reranker: probability-like outputs stay bounded; scalar/logit outputs such as
  `Alibaba-NLP/gte-multilingual-reranker-base` use a monotonic sigmoid mapping;
- `final_score` is the canonical API score and `final_score_source` identifies
  `dense`, `sparse`, `fusion`, or `rerank`;
- raw backend/model values are preserved only in opt-in technical traces and
  omitted from ordinary `RetrievalResult.to_dict()` candidate serialization.

Streamlit shows the clean `[0,1]` final/stage relevance scores first and keeps
raw scores/large metadata inside a collapsed technical inspector.

## Real-data functional checks performed here

The supplied `/mnt/data/45.docx` was parsed with the real DOCX parser and passed
through the updated chunker. It produced content-bearing section chunks such as
`Mục tiêu của đề tài`, `Kết quả đạt được`, and
`Tính mới, tính khoa học và ứng dụng`, with no short standalone heading chunk.

The exact normalized-unit trace supplied for `CD Q3 - 2026.docx` was replayed
through the updated chunker. The resulting chunks had:

- no standalone searchable `Đặt vấn đề`, `Thực trạng`, `Khái niệm`, or
  `Giải pháp, biện pháp khắc phục` title;
- no `Từ nhà trườngMôi trường`,
  `Từ xã hội và môi trường làm việcBên cạnh`, or `Tiểu kếtCó thể` joins;
- no below-minimum chunk in that replay;
- explicit paths including `Thực trạng > Nguyên nhân > Từ nhà trường` and
  `Giải pháp, biện pháp khắc phục > Đối với cá nhân`;
- solution chunks whose text contains real solution content, not only the
  seven-token parent heading.

## Verification executed in this environment

```text
python -m pytest -q \
  rag/tests/unit/test_normalization_chunking.py \
  rag/tests/unit/test_retrieval_semantics.py \
  rag/tests/unit/test_config.py \
  rag/tests/unit/test_sparse_analyzer.py \
  rag/tests/parser
116 passed

python -m compileall -q rag
PASS

python -m rag.verification
RAG runtime isolation: PASS
```

The uploaded repository intentionally omitted `rag/.venv`, and this execution
environment does not contain `streamlit`, `qdrant-client`,
`sentence-transformers`, or `transformers`. Therefore this pass did **not**
claim a fresh execution of the real E5 model, the real Alibaba GTE reranker,
local Qdrant, or Streamlit AppTest. Those gates must be re-run in the project's
existing Windows `rag/.venv` after replacing the source.

## Required reindex after installing this V2 source

Existing `.rag_runtime` vectors/FTS rows were created from the previous chunk
texts and IDs. Updating source code does not rewrite an existing knowledge
base. Delete/reingest or force-reindex affected documents before evaluating the
new retrieval behavior. Clearing `.rag_runtime` is also valid only when it is
acceptable to delete all local indexed knowledge.

## FastAPI service pass

Added a thin HTTP adapter under `rag/api/` without duplicating RAG logic.
The service owns transport validation/serialization only and calls the same
`RagModule`/`KnowledgeService` facade used by the inspection UI.

Key decisions:

- FastAPI lifespan creates one long-lived `RagModule` and closes it on shutdown.
- File ingestion passes FastAPI's spooled file handle into the bounded core
  intake instead of unconditionally reading the whole upload into RAM.
- Normal retrieval responses return compact context/provenance plus one
  canonical relevance `score` in `[0, 1]`; normalized stage scores and raw
  backend/model scores are debug-only.
- `RAG_DATA_DIR` now anchors local Qdrant/FTS5/registry persistence. Source
  checkouts default to repository-root `.rag_runtime`; installed wheels fall
  back to a user-owned home directory. Runtime paths no longer change merely
  because Streamlit/Uvicorn was launched from another working directory.
- The API imports without loading Qdrant, SentenceTransformers, Transformers,
  E5, or GTE. Heavy initialization happens only in service lifespan startup.
- Local embedded Qdrant remains a one-process deployment mode; the CLI starts
  Uvicorn with one worker.

Functional HTTP endpoints cover liveness/readiness, file/text ingestion,
content-aware update, forced reindex, list/detail/chunk inspection, delete,
recovery, and retrieval.

Verification in the artifact-building environment:

```text
python -m compileall -q rag                         PASS
python -m rag.verification                          PASS
pytest API + unit + parser                          140 passed, 1 skipped
pytest rag/tests/api                                9 passed, 1 skipped
```

The skipped API integration test requires `qdrant_client`, which is not
installed in the artifact-building environment. It is intentionally present so
that the repository's full `rag/.venv` executes an actual HTTP -> RagModule ->
local Qdrant/FTS5 flow. The existing real E5/GTE core E2E test remains the gate
for real model execution.
