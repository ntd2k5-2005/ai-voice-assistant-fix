# RAGFlow reference audit

This audit records how the standalone `rag` package used the bundled
`ragflow-main/` checkout as an engineering reference. It is not a claim of
feature parity and it does not make RAGFlow a runtime dependency.

The inspected RAGFlow files carry Apache-2.0 headers; see
`ragflow-main/LICENSE` and `rag/THIRD_PARTY_NOTICES.md`. The line areas below
refer to the bundled checkout at the time of this audit and may move if that
reference tree is replaced.

## Decision and status vocabulary

Decisions:

- **KEEP**: retain a useful contract or invariant in the standalone design.
- **ADAPT**: use the engineering idea but change it for this package's narrower
  local-runtime, typed, testable boundary.
- **REIMPLEMENT**: build an independent implementation under `rag/`; do not
  import the reference implementation.
- **REMOVE**: delete or explicitly reject a misleading/duplicated path.
- **OUT-OF-SCOPE**: deliberately exclude the capability from this retrieval
  service.

Statuses:

- **VERIFIED** means a focused automated test is named in the row.
- **IMPLEMENTED-NOT-REAL-VERIFIED** means code exists but evidence uses
  synthetic fixtures/controlled models or an external environment has not been
  established across the claimed family.
- **OUT-OF-SCOPE** means no product implementation is promised.
- **KNOWN LIMIT** qualifies a narrower verified result.

No benchmark value appears here. Passing a contract test is not retrieval
quality, OCR accuracy, or RAGFlow parity.

## Isolation and composition

| Capability | RAGFlow reference (exact local path/area) | Decision | Standalone target and evidence | Status / divergence |
|---|---|---|---|---|
| Licensing and provenance | `ragflow-main/LICENSE`; Apache headers at `deepdoc/parser/pdf_parser.py:1-14`, `rag/llm/embedding_model.py:1-14`, and `rag/nlp/search.py:1-14` | **KEEP** attribution; **REIMPLEMENT** product code | `rag/THIRD_PARTY_NOTICES.md`; this audit | VERIFIED as documentation |
| Runtime dependency boundary | RAGFlow parsers import its `deepdoc`, vision, model, and service stacks; examples include `deepdoc/parser/pdf_parser.py:42-90` and `rag/app/naive.py:25-64` | **REMOVE** all runtime imports and path injection | `rag/verification.py`; `rag/tests/unit/test_dependency_isolation.py` | VERIFIED; `ragflow-main/` is reference-only |
| Central parser configuration | The `parser_config` dictionary and defaults are interpreted inside `rag/app/naive.py:952-978` and repeatedly read across `1006-1331` | **ADAPT** into immutable typed sections; remove hidden mutation/defaults | `rag/config.py`; factory propagation in `rag/pipeline.py`; `rag/tests/unit/test_config.py`; parser propagation test | VERIFIED |
| Format dispatch | Exported parser families are listed at `deepdoc/parser/__init__.py:17-38`; extension dispatch is in `rag/app/naive.py:1006-1245` | **KEEP** the explicit family map; **REIMPLEMENT** signature-aware dispatch | `rag/ingestion/parsers/registry.py`; parser intake tests | VERIFIED |
| Monolithic parse/chunk entry | `rag/app/naive.py:952-1331` dispatches formats, merges, tokenizes, enriches media, and returns chunks in one path | **REMOVE** that cross-responsibility shape; **ADAPT** only useful ordering/config ideas | `ParserRegistry` -> `normalize_document` -> `UltimateChunker` -> lifecycle in `rag/pipeline.py` | VERIFIED through focused layers and pipeline tests |
| Canonical application facade | No selected RAGFlow file represents this package's narrow two-flow facade; `rag/app/naive.py:952-1331` is the closest ingestion boundary | **REIMPLEMENT** one facade | `rag/service.py` (`RagModule`, `KnowledgeService`); `rag/tests/integration/test_public_service.py` | VERIFIED |

## Parsing and document structure

| Capability | RAGFlow reference (exact local path/area) | Decision | Standalone target and evidence | Status / divergence |
|---|---|---|---|---|
| Plain and source-code text | `deepdoc/parser/txt_parser.py:29-62` splits/merges text by token budget; extensions are routed at `rag/app/naive.py:1133-1139` | **ADAPT** decoding and ordered-line semantics; chunk later in one canonical chunker | `rag/ingestion/parsers/text.py`; registry code-extension map; structured parser and chunking tests | VERIFIED |
| Markdown structure | Fence-safe table handling and structural element extraction are at `deepdoc/parser/markdown_parser.py:32-157,165-567`; integration is at `rag/app/naive.py:1140-1196` | **ADAPT** headings/lists/tables/fences/links without vision enrichment | `rag/ingestion/parsers/markdown.py`; `rag/tests/parser/test_structured_parsers.py`; chunking table tests | VERIFIED on fixtures; image caption enrichment excluded |
| HTML structure | Block tags, ordered extraction, and table splitting are at `deepdoc/parser/html_parser.py:37-218`; dispatch is at `rag/app/naive.py:1197-1204` | **ADAPT** ordered semantic blocks; add executable-noise removal | `rag/ingestion/parsers/html.py`; structured parser tests | VERIFIED on fixtures |
| JSON and JSONL | JSON splitting, falsey/empty handling, and JSONL detection are at `deepdoc/parser/json_parser.py:28-189`; dispatch is at `rag/app/naive.py:1213-1219` | **REIMPLEMENT** record/path-preserving units with fail-whole-input corruption semantics | `rag/ingestion/parsers/json_data.py`; JSONL parser tests | VERIFIED |
| EPUB reading order | Spine traversal and XHTML fallback order are at `deepdoc/parser/epub_parser.py:36-139`; dispatch is at `rag/app/naive.py:1205-1211` | **REIMPLEMENT** spine-first ordered units and item provenance | `rag/ingestion/parsers/epub.py`; EPUB fixture test | VERIFIED |
| DOCX | Paragraph styles and table extraction are at `deepdoc/parser/docx_parser.py:33-181`; document dispatch/merge is at `rag/app/naive.py:1006-1047` | **REIMPLEMENT** body-order walk, headings, tables, and merged-cell provenance | `rag/ingestion/parsers/docx.py`; DOCX structured parser test | VERIFIED on generated OOXML fixtures |
| XLSX/CSV | Workbook/sheet/row extraction and bounded table chunks are at `deepdoc/parser/excel_parser.py:29-310`; routing is at `rag/app/naive.py:1102-1132` | **REIMPLEMENT** sheets, sparse rows, merges/formulas and separate CSV/TSV parsing | `rag/ingestion/parsers/xlsx.py`, `csv.py`; structured/integration metadata tests | VERIFIED on generated fixtures |
| PPTX | Geometric shape order, table extraction, and group recursion are at `deepdoc/parser/ppt_parser.py:22-98` | **REIMPLEMENT** visual order, tables, notes, and slide provenance | `rag/ingestion/parsers/pptx.py`; PPTX structured parser test | VERIFIED on generated fixtures |
| Legacy `.doc`/`.xls`/`.ppt` | RAGFlow optionally uses Tika for `.doc` at `rag/app/naive.py:1221-1245`; its spreadsheet expression includes legacy `.xls` at `1102-1132` | **REMOVE** ambiguous optional legacy support; fail explicitly | `ParserConfig.allowed_extensions`; parametrized unsupported-format tests for all three extensions | VERIFIED |
| File signatures and mismatch policy | The selected RAGFlow dispatch is primarily filename based at `rag/app/naive.py:1006-1245` | **ADAPT** with PDF magic and OOXML/EPUB container inspection plus strict/permissive mismatch policy | `rag/ingestion/parsers/base.py`, `registry.py`; signature/mismatch tests | VERIFIED |
| Bounded byte sources | No selected parser registry path exposes this standalone contract directly; `rag/app/naive.py:952-1004` receives optional binary content | **REIMPLEMENT** bounded temporary materialization with deterministic identity and no leaked temp path | `BytesParser` in `rag/ingestion/parsers/registry.py`; byte-source tests | VERIFIED |
| Typed parser failures | RAGFlow paths often log, return empty output, or raise backend-specific exceptions (for example `.doc` at `rag/app/naive.py:1220-1245`) | **REIMPLEMENT** machine-readable failure categories | `rag/ingestion/parsers/base.py`, `rag/errors.py`; parser error tests | VERIFIED |

## PDF, layout, OCR, and tables

| Capability | RAGFlow reference (exact local path/area) | Decision | Standalone target and evidence | Status / divergence |
|---|---|---|---|---|
| Native and plain PDF paths | `RAGFlowPdfParser` begins at `deepdoc/parser/pdf_parser.py:56`; image/page extraction and main call are at `1608-1769`; `PlainParser` is at `2071-2126` | **REIMPLEMENT** a smaller per-page pdfplumber path | `rag/ingestion/parsers/pdf.py`; normal/corrupt/encrypted/empty/page-limit tests | VERIFIED on synthetic fixtures; not full DeepDOC parity |
| Garbled-text detection and OCR fallback | Character/font-encoding heuristics are at `deepdoc/parser/pdf_parser.py:199-365`; box/page fallback is used at `825-865` and `1630-1716` | **ADAPT** per-page native-usability checks plus explicit backend selection or ordered Tesseract -> RapidOCR auto fallback | PDF OCR methods; scanned, mixed-page, and auto-fallback tests | Ordering, selected-backend metadata, aggregated all-backend failure diagnostics, and Tesseract fixture path VERIFIED; broader scripts are a KNOWN LIMIT |
| Column/layout order | Layout recognition/global columns are at `deepdoc/parser/pdf_parser.py:877-969` | **REIMPLEMENT** deterministic word geometry and basic two-column split; retain line boxes | `PDFParser._native_lines`, `_column_split`, `_words_to_lines`; multicolumn test | VERIFIED for the generated two-column case; complex layout IMPLEMENTED-NOT-REAL-VERIFIED |
| Tables | Rotation/table-structure OCR is at `deepdoc/parser/pdf_parser.py:367-769`; table/figure extraction is at `1287-1607` | **REIMPLEMENT** local pdfplumber table detection and separate Markdown table units | `PDFParser._tables`; PDF table fixture; table-aware chunk tests | Basic ruled-table path VERIFIED; rotation/semantic TSR not implemented |
| Repeated page margins | RAGFlow's broader layout/text merge is at `deepdoc/parser/pdf_parser.py:877-1040` | **ADAPT** repeated normalized top/bottom line removal with diagnostics | `PDFParser._remove_repeated_margins`; repeated header/footer test | VERIFIED on fixture |
| OCR backends | RAGFlow owns a learned OCR/layout stack at `deepdoc/parser/pdf_parser.py:42-90,1608-1716` | **REIMPLEMENT** host Tesseract plus optional RapidOCR; do not ship DeepDOC models | `PDFParser._ocr_page`, `_ocr_tesseract`, `_ocr_rapid`; `OCRConfig`; PDF parser tests | Auto control flow/diagnostics and Tesseract fixture path VERIFIED; real RapidOCR runtime/model breadth IMPLEMENTED-NOT-REAL-VERIFIED |
| Alternative PDF engines and vision enrichment | `rag/app/naive.py:1049-1123` selects DeepDOC/plain/TCADP/Docling/MinerU/PaddleOCR/OpenDataLoader/SoMark/Mistral paths; Markdown figures are enriched at `1150-1187` | **OUT-OF-SCOPE** | No provider API, vision caption model, or alternate parser service in `rag` | OUT-OF-SCOPE |

## Normalization, chunking, and identity

| Capability | RAGFlow reference (exact local path/area) | Decision | Standalone target and evidence | Status / divergence |
|---|---|---|---|---|
| Token-budget merging | Parser chunk limits/delimiters are configured at `rag/app/naive.py:952-978`; Markdown and general merging are at `1247-1307`; plain text merging is at `deepdoc/parser/txt_parser.py:35-62` | **ADAPT** one tokenizer-aware, model-limit-aware chunker after all parsers | `rag/processing/chunking.py`; normalization/chunking tests | VERIFIED |
| Structure preservation | RAGFlow keeps tables separate and protects Markdown fences/tables (`deepdoc/parser/markdown_parser.py:36-157,288-567`) | **KEEP** the invariant; **REIMPLEMENT** logical units/table-row split | `rag/processing/clean_text.py`, `chunking.py`; table, code, heading, Unicode tests | VERIFIED |
| Heading continuity | RAGFlow's Markdown block/header logic is at `deepdoc/parser/markdown_parser.py:288-426` | **ADAPT** heading transition flush and optional heading prefix, including Vietnamese labels | `UltimateChunker._logical_units`; heading-transition test | VERIFIED |
| Overlap | RAGFlow reads overlap and merges at `rag/app/naive.py:1247-1307` | **ADAPT** overlap by the active token counter and expose actual overlap metadata | `UltimateChunker`; long paragraph/limit tests | VERIFIED |
| Stable chunk identity | RAGFlow emits chunk metadata/order through tokenization near `rag/app/naive.py:1293-1331`; no selected area provides this package's versioned hash contract | **REIMPLEMENT** deterministic content/provenance/version identity | `UltimateChunker._create_payload`; identity tests; cross-store manifest checks | VERIFIED |
| Destructive generic cleaning | No reference behavior is adopted that would flatten every structure before chunking | **REMOVE** from the product path | `clean_text` preserves tables/lists/code/page separators and metadata | VERIFIED |

## Embedding and reranking models

| Capability | RAGFlow reference (exact local path/area) | Decision | Standalone target and evidence | Status / divergence |
|---|---|---|---|---|
| Separate document/query embedding API | RAGFlow's base contract declares `encode()` and `encode_queries()` at `rag/llm/embedding_model.py:147-161`; provider implementations repeat the separation (for example `223-281`, `584-607`, `1334-1405`) | **KEEP** the semantic separation; **ADAPT** names to `embed_documents()`/`embed_query()` | `rag/Embeddings/embeddings.py`, adapters/factory; embedding capability and benchmark tests | Contract VERIFIED |
| Ordered batching and consistent failures | Shared batching/order/error handling is at `rag/llm/embedding_model.py:51-64,162-220` | **ADAPT** for local SentenceTransformer, progress, dimension checks, and batch-halving OOM recovery | `EmbeddingEngine`; model capability tests | VERIFIED with controlled models |
| E5/BGE/Qwen/generic semantics | RAGFlow identifies local built-in maximums and document/query calls at `rag/llm/embedding_model.py:223-281` and exposes numerous provider-specific paths throughout the file | **REIMPLEMENT** explicit local capability profiles and override validation | capability tests; real public-facade E2E | Family contract and default E5 CPU execution VERIFIED; BGE/Qwen/Nomic/private families and CUDA IMPLEMENTED-NOT-REAL-VERIFIED |
| Dimension/max length/checkpoint/profile identity | RAGFlow truncates provider inputs in the shared batch path at `rag/llm/embedding_model.py:163-212` | **REIMPLEMENT** introspection/probe, configured fallback source, concrete loaded revision resolution, effective profile hash, mismatch failure, and stored index identity | `rag/model_identity.py`; engine `resolved_config`; Qdrant identity tests; raw model artifacts | VERIFIED; revision is recorded when model/tokenizer metadata exposes it |
| Remote code/offline/revision policy | The selected RAGFlow provider stack has different deployment/security assumptions across `rag/llm/embedding_model.py` | **REIMPLEMENT** explicit `trust_remote_code`, requested `revision`, and `local_files_only`; never infer trust | embedding/reranker config and capability tests | VERIFIED; runtime records the loaded commit when available, but operators remain responsible for pinning an immutable reviewed revision |
| Reranker score normalization | RAGFlow's common `similarity()` contract and `[0,1]` normalization are at `rag/llm/rerank_model.py:35-95`; downstream model blend is at `rag/nlp/search.py:494-525` | **ADAPT** profile-specific extraction while preserving raw rerank score separately from fusion | `rag/retrieval/reranking.py`, `reranker_capabilities.py`; reranker tests | Contract VERIFIED |
| Sequence classifier families | RAGFlow contains provider and Hugging Face rerank adapters throughout `rag/llm/rerank_model.py`, including local HTTP batching at `666-718` | **REIMPLEMENT** exact pinned Alibaba GTE raw-scalar adapter plus generic/BGE/Jina conventions with resolved revision/profile identity | reranker capability tests and real public-facade E2E | Required Alibaba GTE CPU path VERIFIED; other checkpoints and CUDA IMPLEMENTED-NOT-REAL-VERIFIED |
| Qwen reranking | RAGFlow's Qwen provider adapter is at `rag/llm/rerank_model.py:626-665` | **ADAPT** for local Qwen3 causal yes/no probability rather than provider relevance output | Qwen profile and causal adapter; Qwen controlled-model test | IMPLEMENTED-NOT-REAL-VERIFIED with actual Qwen weights |
| Cloud model providers | RAGFlow's embedding/rerank files implement many hosted SDK/HTTP providers | **OUT-OF-SCOPE** for this local standalone package | No OpenAI/DashScope/Cohere/Voyage/etc. service adapter in the canonical factory | OUT-OF-SCOPE |

## Dense, sparse, fusion, retrieval, and lifecycle

| Capability | RAGFlow reference (exact local path/area) | Decision | Standalone target and evidence | Status / divergence |
|---|---|---|---|---|
| Dense + lexical candidate construction | RAGFlow builds text/dense expressions and weighted fusion at `rag/nlp/search.py:121-241` | **ADAPT** into independent dense and sparse branches with an explicit mode | `rag/retrieval/dense.py`, `bm25.py`, `hybrid.py`; retrieval semantics tests | VERIFIED |
| Active visibility | RAGFlow passes `available_int=1` at `rag/nlp/search.py:575-599` and prunes orphaned documents at `64-118,600-607` | **KEEP** fail-closed visibility; **REIMPLEMENT** registry CAS plus shared retrieval visibility epochs | store snapshot APIs; atomic-visibility and cross-CAS hybrid tests | VERIFIED without zero-visible or mixed-version result windows |
| Metadata filters | RAGFlow request filters are formed at `rag/nlp/search.py:121-133` | **ADAPT** a backend-neutral typed predicate set with dense/sparse parity | `MetadataFilter`; Qdrant/FTS filter compilation; parity and hardening tests | VERIFIED |
| Weighted fusion | RAGFlow uses weighted search/rerank blends at `rag/nlp/search.py:199-226,434-525` | **ADAPT** weighted blending over already-normalized public branch relevance, active-weight renormalization, and preserved raw backend scores | `WeightedScoreFusion`; retrieval semantics tests | VERIFIED; optional strategy, not default |
| Reciprocal rank fusion | No RRF path was selected in the inspected `rag/nlp/search.py` areas | **REIMPLEMENT** rank-only default for dense/BM25 scale robustness; expose a theoretical-maximum-normalized `[0,1]` public relevance while retaining raw RRF in trace | `ReciprocalRankFusion`; equal/negative/empty/tie tests | VERIFIED |
| Candidate pool vs returned Top-K | RAGFlow bounds a rerank/pagination window at `rag/nlp/search.py:525-599` and slices at `668-691` | **KEEP** the separate pool/output concepts; **ADAPT** to explicit `candidate_k >= top_k` | `RetrievalEngine.retrieve`; propagation/window tests | VERIFIED |
| Optional reranker and truthful fallback | RAGFlow switches model rerank at `rag/nlp/search.py:619-657` | **ADAPT** independent rerank flag for every mode, explicit timeout, fallback-or-raise policy, diagnostics | `rag/retrieval/hybrid.py`; unit/integration fallback tests | VERIFIED |
| Score/provenance result contract | RAGFlow assembles chunk fields at `rag/nlp/search.py:668-738` | **REIMPLEMENT** typed candidates retaining branch/fusion/rerank/final scores and stage latency | `rag/contracts.py`; retrieval semantics/E2E tests | VERIFIED |
| Embedding/index identity | No selected RAGFlow search area provides this local collection manifest contract | **REIMPLEMENT** model/dimension/normalization identity and mismatch fail-fast | `rag/storage/qdrant.py`; Qdrant identity tests | VERIFIED |
| Dual-store staged activation | RAGFlow's availability/orphan guards at `rag/nlp/search.py:64-133,575-607` motivate fail-closed reads but do not provide this local Qdrant+SQLite protocol | **REIMPLEMENT** stage/read-back parity, target physically ready while old stays readable, registry CAS, then cleanup | coordinator/recovery; lifecycle phase, atomic-visibility, and real rollback tests | VERIFIED |
| Restart recovery | No equivalent selected RAGFlow file is reused | **REIMPLEMENT** pending ingest/delete/cleanup compensation on startup | `rag/lifecycle/recovery.py`; recovery and crash-phase tests | VERIFIED |
| Operation idempotency/concurrency | No equivalent selected RAGFlow file is reused | **REIMPLEMENT** stable operation IDs, ownership, SQLite `BEGIN IMMEDIATE`, CAS commit | `rag/lifecycle/manager.py`; operation replay and multiprocess owner tests | VERIFIED for registry ownership; local Qdrant sharing is a KNOWN LIMIT |
| Distributed stores/multi-tenancy | RAGFlow search targets multiple document engines, tenant indexes, and knowledge bases at `rag/nlp/search.py:134-241,554-607` | **OUT-OF-SCOPE** | Canonical stores are embedded local Qdrant and SQLite FTS5 | OUT-OF-SCOPE |

## Facade, UI, benchmarks, and test evidence

| Capability | RAGFlow reference (exact local path/area) | Decision | Standalone target and evidence | Status / divergence |
|---|---|---|---|---|
| Knowledge lifecycle API | RAGFlow's inspected `naive.chunk()` path is ingestion-oriented at `rag/app/naive.py:952-1331` | **REIMPLEMENT** typed add/update/reindex/delete/recover/inspect operations | `rag/service.py`, `rag/pipeline.py`; public-service and lifecycle tests | VERIFIED |
| Optional UI | No inspected RAGFlow parser/search area is adopted as a UI contract | **ADAPT** local Streamlit to call only cached `RagModule`; **REMOVE** duplicate orchestration/tunables | public-facade AppTests with real local stores; compile/import/isolation; headless health HTTP 200 | VERIFIED; manual cross-browser rendering remains environment-specific |
| Model benchmark | No selected reference path supplies the standalone fixed labelled evaluation contract | **REIMPLEMENT** separate document/query embedding evaluation and fixed reranker pools | historical benchmark artifacts and framework tests | Framework VERIFIED; old MiniLM/MS-MARCO results are not current completion evidence |
| E2E benchmark | No selected reference path is reused | **REIMPLEMENT** labelled mode comparison with raw rankings and timing | historical benchmark artifacts and framework tests | Historical narrow smoke only; no new quality benchmark was required or run for completion |
| Failure injection | No selected reference path supplies this local protocol matrix | **REIMPLEMENT** deterministic crash checkpoints around registry/write/verify/activate/cleanup/delete | `StorageLifecycleCoordinator.CHECKPOINTS`; recovery tests | VERIFIED |
| Real-model smoke | RAGFlow reference uses a materially broader service/model stack | **ADAPT** one cached-only public-facade vertical test | `rag/tests/e2e/test_real_pipeline.py` | Default E5 + exact pinned required Alibaba GTE + real parser/stores/lifecycle CPU path VERIFIED; no broad quality/family/CUDA claim |
| LLM answer generation, agents, memory, STT/TTS, GraphRAG | These exist elsewhere in the full RAGFlow product but are outside the inspected retrieval engineering subset | **OUT-OF-SCOPE** | Not imported, configured, tested, or exposed by `RagModule` | OUT-OF-SCOPE |

## Resulting differences from RAGFlow

The standalone module intentionally does less:

- no DeepDOC learned layout/OCR/table-structure stack and no alternate cloud or
  service PDF parsers;
- no vision captioning, image retrieval, hyperlink crawling, or recursive
  embedded-document enrichment;
- no Tika fallback for legacy Office formats;
- no hosted embedding/reranker provider matrix;
- no Elasticsearch/Infinity/OceanBase/SereneDB abstraction, multi-tenant
  knowledge-base service, answer LLM, or agent runtime.

It adds narrower invariants needed for a verifiable local service:

- immutable central configuration and dependency-light typed contracts;
- deterministic parser failures, structured provenance, chunk identity, and
  explicit model capability/security policy with resolved checkpoint/profile
  evidence;
- dense/sparse manifest parity, operation ownership, fail-closed version
  visibility, compensation, and restart recovery;
- explicit retrieval modes, separate candidate/output windows, weighted or RRF
  fusion, truthful reranker fallback, and per-stage scores/latency;
- raw benchmark artifacts generated from one evidence object rather than
  hand-authored metrics.

## Verification boundary

Run the repository's focused suites from `rag-service` with `rag/.venv`; do not
infer the statuses above from documentation alone:

```powershell
rag\.venv\Scripts\python.exe -m rag.verification
rag\.venv\Scripts\python.exe -m pytest rag\tests\parser -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\model -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\integration rag\tests\recovery -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\e2e\test_real_pipeline.py -q
```

For optional OCR or additional real-model paths, record the exact
executable/languages, requested and resolved revisions, profile hash, cache
policy, device/dtype, configuration snapshot, failures, and skips. The checked-in
artifacts establish the named cached CPU models only; other families and CUDA
remain unverified. Only a raw benchmark artifact with reviewed labels may
support a retrieval-quality comparison.
