# RAG service hardening audit

This report audits the deployable `rag` package in the `rag-service`
repository. It is an architecture and evidence map, not a retrieval-quality
claim. Quality measurements belong in machine-generated benchmark artifacts.

## Status vocabulary

- **VERIFIED**: the implementation has focused automated evidence named in
  this report. It does not imply parity on every production document/model.
- **IMPLEMENTED-NOT-REAL-VERIFIED**: the path exists, but evidence is synthetic,
  uses controlled model doubles, or depends on an external runtime/checkpoint
  that has not been established for every deployment target.
- **OUT-OF-SCOPE**: deliberately excluded from this retrieval-only module.
- **KNOWN LIMIT**: an explicit production boundary, not silently presented as
  complete support.

The test suite remains the authority at a particular commit. This document
does not freeze a pass count because adding parametrized cases would make that
number stale without changing the covered invariant.

## Audited boundary

Only `rag/` is product runtime. `ragflow-main/` is a read-only Apache-2.0
reference. The package must have no import from `ragflow`, `ragflow_main`, or
`deepdoc`, no runtime `sys.path` mutation, and no library-owned
`logging.basicConfig()` call. `rag.verification` statically enforces those
rules; `rag/tests/unit/test_dependency_isolation.py` exercises the scanner.

The module supports exactly two application flows:

1. knowledge ingestion/management: parse, normalize, chunk, embed, stage,
   verify, activate, update/reindex, inspect, recover, and delete;
2. retrieval: validate, dense/sparse/hybrid search, fuse, optionally rerank,
   and return typed candidates with provenance, scores, latency, and diagnostics.

Answer generation, prompting, STT, TTS, memory, agents, and GraphRAG are not
part of this runtime.

## Canonical runtime map

| Responsibility | Canonical location | Boundary |
|---|---|---|
| Configuration | `rag/config.py` | Immutable nested `RagConfig`; JSON-compatible snapshot; constructor validation |
| Public contracts/errors | `rag/contracts.py`, `rag/errors.py` | Dependency-light source, parsed unit, chunk, result, filter, ingestion, and typed error contracts |
| Public facade | `rag/service.py` | One `RagModule`; `KnowledgeService` for lifecycle operations |
| Composition | `rag/pipeline.py` | Only production parser/model/store/lifecycle assembly; startup recovery |
| Parsing | `rag/ingestion/parsers/` | Signature-aware registry and format-specific structured parsers |
| Normalization/chunking | `rag/processing/` | Unicode/structure-preserving normalization and tokenizer-aware deterministic chunks |
| Embeddings/model identity | `rag/Embeddings/`, `rag/model_identity.py` | Long-lived engine, capability profiles, document/query adapters, resolved checkpoint/profile identity, OOM recovery |
| Dense store | `rag/storage/qdrant.py` | Local Qdrant, embedding identity, active-state gate, staged manifests |
| Sparse store | `rag/storage/fts5.py` | SQLite FTS5, JSON metadata, filters, active-state gate, staged manifests |
| Lifecycle | `rag/lifecycle/` | Registry ownership/CAS, dual-store coordinator, compensation, restart recovery |
| Retrieval | `rag/retrieval/` | Dense/BM25 branches, weighted/RRF fusion, reranker profiles, truthful fallback |
| Optional UI | `rag/streamlit_app/` | Thin cached `RagModule` adapter only; no second ingestion/retrieval implementation |
| Evaluation | `rag/benchmark.py`, `rag/metrics.py` | Model and E2E metrics, raw rankings/timing, JSON/CSV/generated Markdown |

`Embeddings` remains capitalized as the single canonical package because a
second case-only spelling is unsafe on Windows filesystems.

## Decision and verification matrix

| Area | Decision | Current result | Evidence | Status |
|---|---|---|---|---|
| Central configuration | **KEEP** one nested immutable configuration; remove inert duplicated knobs | Paths, parser/OCR, normalization, chunking, models, stores, fusion, retrieval, and lifecycle propagate from `RagConfig`; invalid combinations fail fast | `rag/tests/unit/test_config.py`; parser propagation in `rag/tests/parser/test_structured_parsers.py` | VERIFIED |
| Public boundary | **KEEP** `RagModule`; **REMOVE** application-level direct assembly as the normal path | Knowledge and retrieval are separate facade methods over one long-lived pipeline | `rag/service.py`; `rag/tests/integration/test_public_service.py` | VERIFIED |
| Reference isolation | **REMOVE** all RAGFlow imports/path tricks | Runtime is independently deployable; RAGFlow is documentation/reference only | `rag.verification`; `rag/tests/unit/test_dependency_isolation.py` | VERIFIED |
| Parser intake | **ADAPT** content-aware dispatch; reject ambiguity according to policy | File/text/bytes/upload sources, bounded cursor-safe reads, size/unit limits, signatures, strict mismatch option, ZIP traversal/encryption/decompression preflight, typed parser diagnostics | parser source/archive/structured tests | VERIFIED |
| Structured formats | **REIMPLEMENT** locally | DOCX/XLSX/PPTX/HTML/Markdown/JSON/JSONL/EPUB/plain/code units retain ordered structure and provenance | `rag/tests/parser/test_structured_parsers.py`; `rag/tests/integration/test_structured_metadata.py` | VERIFIED on generated fixtures |
| Legacy Office | **REMOVE** false support | `.doc`, `.xls`, `.ppt` fail explicitly; callers must convert to OOXML | `ParserConfig.allowed_extensions`; parametrized legacy-format tests | VERIFIED |
| PDF native extraction | **REIMPLEMENT** a lighter local path | Per-page extraction, repeated-margin removal, basic two-column order, separate tables, page/layout provenance, corrupt/encrypted/empty limits | `rag/tests/parser/test_pdf_parser.py` | VERIFIED on synthetic PDFs; KNOWN LIMIT for broad production layout |
| PDF OCR | **ADAPT** per-page fallback | Explicit backend selection or ordered auto fallback Tesseract -> RapidOCR; method/confidence retained; all-backend failure diagnostics name every attempted backend | real Tesseract, mixed-page, and auto-fallback tests in `rag/tests/parser/test_pdf_parser.py` | Ordering/diagnostics and Tesseract fixture path VERIFIED; real RapidOCR backend breadth IMPLEMENTED-NOT-REAL-VERIFIED |
| Normalization | **KEEP** structural whitespace and provenance; **REMOVE** destructive cleaning | Idempotent Unicode/control/line normalization without flattening tables, lists, code, or page boundaries | `rag/tests/unit/test_normalization_chunking.py` | VERIFIED |
| Chunking | **ADAPT** structure-first grouping to the active tokenizer/model limit | Ordered cross-unit heading paths, no standalone heading-only knowledge chunks, paragraph/line/sentence-aware overlap, explicit separators, row-safe tables, oversized-block fallback, deterministic version-sensitive IDs | `rag/tests/unit/test_normalization_chunking.py` | VERIFIED in dependency-light tests; real-model reindex required for existing knowledge |
| Embedding semantics | **ADAPT** separate passage/query paths | E5/E5-instruct/BGE/BGE-M3/Qwen/Nomic/generic profiles, explicit overrides, stable batching, dimension/max-length validation, resolved revision/profile hash, normalization identity, OOM reduction | model capability tests; real public-facade E2E | Contract and default E5 CPU execution VERIFIED; other families/CUDA IMPLEMENTED-NOT-REAL-VERIFIED |
| Reranker semantics | **ADAPT** model-specific input/score extraction | Required pinned Alibaba GTE raw-scalar adapter plus generic/BGE/Jina/Qwen3 profiles; independent weight/code revision identity; deterministic ties; OOM retry; timeout/failure policy | model/retrieval tests; real public-facade E2E | Contract and exact pinned Alibaba GTE CPU execution VERIFIED; other checkpoints/CUDA IMPLEMENTED-NOT-REAL-VERIFIED |
| Dense identity/filtering | **KEEP** cosine retrieval; **REIMPLEMENT** persistent identity and visibility checks | Model/dimension/normalization plus available resolved checkpoint/profile identity fail fast, staged state, registry gate, native filters, complete `contains` fallback | `rag/tests/test_contracts.py`; `rag/tests/integration/test_retrieval_hardening.py` | VERIFIED with local Qdrant |
| Sparse index | **KEEP** SQLite FTS5; **ADAPT** it to lifecycle parity | Persistent incremental rows, JSON metadata, SQL filtering, staged/active/stale state, deterministic larger-is-better score | `rag/tests/integration/test_fts5_metadata.py`; recovery tests | VERIFIED |
| Fusion | **ADAPT** explicit strategy object | RRF is the default for incomparable dense/BM25 scales; weighted normalized-score fusion remains optional; public fusion relevance is bounded to `[0,1]`, raw values remain diagnostic | `rag/tests/unit/test_retrieval_semantics.py` | VERIFIED in dependency-light retrieval tests |
| Retrieval modes | **REMOVE** implicit reranking | `dense`, `sparse`/`bm25`, and `hybrid`; independent `rerank`; exact `candidate_k` vs `top_k`; sparse mode does not embed | `rag/tests/unit/test_retrieval_semantics.py` | VERIFIED |
| Rerank degradation | **KEEP** pre-rerank order as fallback and expose the truth | Absent/timeout/invalid reranker either returns identical pool with explicit diagnostics or raises by policy | unit and integration retrieval hardening tests | VERIFIED |
| Dual-store activation | **REIMPLEMENT** document-atomic visibility | Stage/verify both stores, make target physically ready while old remains searchable, registry CAS as one logical switch, then stale cleanup; hybrid branches share a visibility epoch and retry on CAS | lifecycle protocol, atomic-visibility, and real rollback tests | VERIFIED with local stores, concurrency, and failure injection |
| Recovery/delete/idempotency | **REIMPLEMENT** explicit operation ownership | Stable operation IDs, restart compensation, cleanup retry, delete tombstone/repair, SQLite `BEGIN IMMEDIATE` ownership | `rag/tests/recovery/test_storage_lifecycle_protocol.py`; `test_lifecycle_recovery.py` | VERIFIED; shared local Qdrant multi-worker deployment is a KNOWN LIMIT |
| Public observability | **REIMPLEMENT** bounded opt-in diagnostics | Runtime snapshot; ingestion/retrieval stage traces; active chunk inspection; bounded/redacted JSON-safe metadata and vector previews | public observability and Streamlit tests | VERIFIED |
| Optional Streamlit | **ADAPT** to a thin facade; **REMOVE** duplicate orchestration | Cached long-lived module; all trace/inspect flows call only public facade operations | Streamlit AppTests with real local stores; compile/import/isolation; headless health HTTP 200 | VERIFIED; manual cross-browser rendering remains environment-specific |
| Benchmark framework and checked-in smoke | **KEEP** evaluation separate from completion | Historical E5/MiniLM/MS-MARCO artifacts remain reproducible but are explicitly not current GTE or completion evidence | benchmark unit tests; `rag/BENCHMARK_REPORT.md` | Framework VERIFIED; historical quality numbers are not a module-completion claim |
| Real-model E2E | **KEEP** one vertical smoke path | Public `RagModule`, default E5, exact pinned Alibaba GTE, real Markdown parsing, Qdrant/FTS5, dense/sparse/hybrid/rerank, inspect, update, recovery, delete | `rag/tests/e2e/test_real_pipeline.py` | VERIFIED on the named CPU checkpoints; other families/CUDA not verified by this test |
| Generation and agent features | **OUT-OF-SCOPE** | No answer LLM, prompt template, citations composer, chat/agent state, STT, or TTS | package boundary and dependency scan | OUT-OF-SCOPE |

## Enforced invariants

### Ingestion and provenance

- A durable production ingestion cannot silently downgrade to dense-only or
  ephemeral sparse state.
- Content-aware versioning hashes normalized text plus stable structured
  metadata; volatile timestamps/durations are excluded.
- Every staged chunk belongs to exactly one document, version, and operation.
  Both stores expose matching manifests before registry activation.
- Parser provenance survives normalization, chunking, dense/sparse persistence,
  fusion, and the returned `RetrievalCandidate`.
- Empty/corrupt/encrypted/oversized/unsupported input fails with an inspectable
  category instead of an empty-success response.

### Retrieval

- Query normalization is limited to non-destructive Unicode/outer-whitespace
  handling; the document cleaner is not applied to user queries.
- Documents use `embed_documents()` and queries use `embed_query()` (or the
  ordered query batch equivalent). A sparse-only request does neither.
- Metadata filters execute before Top-K on both stores. Incompatible values
  fail closed; Qdrant `contains` does a complete paginated exact fallback.
- Dense, sparse, fusion, reranker, and final scores are separate fields. Every
  returned score is finite and sort ties resolve by stable chunk identity.
- A requested but unapplied rerank is never reported as successful.

### Lifecycle and resources

- The registry is the final visibility authority. Pending, stale, failed, and
  deleting versions cannot become searchable merely because a physical store
  contains rows.
- Normal exceptions compensate staged writes. Simulated process crashes leave
  enough state for idempotent startup recovery.
- Activation precedes old-version cleanup. A cleanup error keeps the new
  version active and records repair work rather than rolling back good data.
- `close()` releases long-lived reranker, embedder, sparse store, dense client,
  and registry without creating per-query model instances.

## Evidence boundaries and known limitations

The following must not be upgraded to broader claims without new artifacts:

- Generated PDF fixtures prove specific Unicode, scanned, mixed-page,
  two-column, table, encryption, corruption, and limit cases. They do not prove
  accuracy on arbitrary newspapers, academic layouts, rotated tables,
  handwriting, or low-resolution Vietnamese scans.
- Auto OCR fallback order and aggregated all-backend failure diagnostics are
  tested. Tesseract still depends on an external executable/language set, and
  real RapidOCR model/runtime breadth needs target-host testing. Neither path
  is DeepDOC parity.
- The model runtime resolves a concrete loaded commit when model/tokenizer
  metadata exposes one and hashes the effective profile. Qdrant persists the
  embedding revision/profile identity. Operators must still pin an immutable,
  reviewed revision; discovery after loading is evidence, not prevention.
- Cached-only CPU completion evidence covers the default multilingual E5 and
  exact pinned Alibaba GTE checkpoints through the public facade. Historical
  MiniLM/MS-MARCO artifacts remain evaluation records only. Other
  BGE/Qwen/Jina/Nomic/private checkpoints, dtypes, and CUDA remain
  IMPLEMENTED-NOT-REAL-VERIFIED.
- Embedded Qdrant plus local SQLite is a single-service deployment. Registry
  contention tests prove version ownership, not safe horizontal sharing of a
  Qdrant directory.
- Benchmarks are meaningful only with reviewed relevance labels and raw
  artifacts. A smoke test, parser test, or successful retrieval is not Recall,
  MRR, nDCG, faithfulness, or answer quality.
- The optional UI is not a second API contract and must never contain its own
  chunking, index synchronization, fusion, or lifecycle logic.

## Repository hygiene decisions

- Keep one implementation per responsibility and route compatibility only at
  a narrow public boundary (for example the deprecated `hybrid_rerank` alias).
- Remove obsolete parser/model/store wrappers, duplicate Streamlit orchestration,
  hardcoded benchmark scripts, and direct top-level imports that work only when
  the current directory is `rag/`.
- Exclude `.venv`, `.pytest_cache`, `__pycache__`, local Qdrant/SQLite data,
  generated benchmark runs, and notebooks from the deployable package. Before
  a release, check the actual source-control diff; ignore rules are not proof
  that an already tracked artifact disappeared.
- Keep third-party attribution in `rag/THIRD_PARTY_NOTICES.md` and the detailed
  comparison in `rag/RAGFLOW_REFERENCE_AUDIT.md`.

## Release verification

Run from the repository root using only `rag/.venv`:

```powershell
rag\.venv\Scripts\python.exe -m rag.verification
rag\.venv\Scripts\python.exe -m build
rag\.venv\Scripts\python.exe -m pytest rag\tests\unit -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\parser -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\integration rag\tests\recovery -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\model -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\e2e\test_real_pipeline.py -q
rag\.venv\Scripts\python.exe -m pytest -q
```

Record skipped external prerequisites and failures explicitly. Do not replace a
failed real-model/OCR gate with a controlled-double unit test while continuing
to label the real path verified.
