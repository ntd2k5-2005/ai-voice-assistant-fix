# Robust RAG service

`rag` is the deployable package inside the `rag-service` repository. It owns
knowledge ingestion, knowledge lifecycle, and ranked retrieval. It does **not**
generate an answer and does not implement STT, TTS, chat memory, agents, or an
LLM prompt chain.

The bundled `ragflow-main/` tree is an Apache-2.0 engineering reference only.
Production code under `rag/` neither imports it nor adds it to `sys.path`.

## Supported flows

Knowledge flow:

```text
DocumentSource -> detect/parse -> normalize -> token-aware chunk
               -> document embeddings -> stage dense + sparse
               -> verify parity -> activate version -> clean stale version
```

Retrieval flow:

```text
validated query -> dense and/or sparse candidates -> weighted or RRF fusion
                -> optional reranker -> typed Top-K result + provenance
                -> per-stage scores, latency, and fallback diagnostics
```

`RagModule` is the canonical application facade. `RagModule.knowledge` exposes
add/update/reindex/delete/recover/inspect; `RagModule.retrieve()` exposes the
read path. Lower-level classes remain available for tests and explicit
dependency injection, but application code should not assemble a second
production pipeline.

## Installation

Run commands from the repository root, `A:\NCKH\rag-service`. Install only into
the existing `rag/.venv`; do not use a global Python or create another virtual
environment.

```powershell
rag\.venv\Scripts\python.exe -m pip install -e ".[test]"
rag\.venv\Scripts\python.exe -m build
```

Install optional integrations into the same environment when they are needed:

```powershell
rag\.venv\Scripts\python.exe -m pip install -e ".[ocr,ui,api,test]"
```

The `ocr` extra installs RapidOCR's Python runtime. Tesseract is a separate
system program: its executable and configured language data (for example `vie`
and `eng`) must also be installed on the host. With `backend="auto"`, OCR tries
Tesseract first and then the optional RapidOCR backend. The selected
method/confidence is retained on a successful page; if every backend fails, the
diagnostic names each failed backend and the parser fails explicitly rather
than turning a scanned page into an empty document. Selecting `tesseract` or
`rapidocr` directly disables backend fallback.

## Input formats and provenance

The parser registry accepts local files, bounded byte payloads, in-memory text,
and uploaded/file-like objects. File-like reads are size bounded, preserve the
caller cursor when possible, and require a safe basename. File dispatch uses
the declared extension and, where enabled, a content signature/container
check. OOXML and EPUB ZIP containers are preflighted before extraction for
unsafe names, symlinks, encryption, duplicate paths, member count, expanded
size, per-member size, and compression-ratio limits.

| Family | Extensions | Preserved structure |
|---|---|---|
| PDF | `.pdf` | page number, extraction method, OCR flag/confidence, layout lines, columns, repeated-margin diagnostics, separate table units |
| Word | `.docx` | body order, headings, paragraphs, tables, merged-cell provenance |
| Spreadsheet | `.xlsx`, `.xlsm`, `.csv`, `.tsv` | sheet, row, cells/columns, merged ranges, formulas where exposed by the source |
| Presentation | `.pptx` | slide, visual shape order, tables, notes |
| Web/markup | `.html`, `.htm`, `.md`, `.markdown`, `.mdx` | headings, lists, links, code, tables, document order; executable HTML noise is removed |
| Structured data | `.json`, `.jsonl`, `.ldjson` | object/record path and ordered record provenance, including falsey values |
| Books | `.epub` | spine reading order and item provenance |
| Plain/code text | `.txt`, `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.h`, `.php`, `.go`, `.sh`, `.cs`, `.kt`, `.sql` | Unicode, indentation, and line semantics |

Legacy binary Office formats `.doc`, `.xls`, and `.ppt` are intentionally
rejected. Convert them to `.docx`, `.xlsx`, or `.pptx` before ingestion. The
runtime does not pretend that an OOXML parser can safely parse OLE files.

Every emitted unit and chunk retains a `document_id`, source, source part, and
format-specific fields such as page, section/heading, sheet/row, slide, table,
or bounding box when available. Chunk IDs are deterministic over logical
position, normalized content, provenance, and document version.

### Structure-aware chunk construction

Chunking consumes the **ordered normalized parser units as one document
sequence**, rather than independently chunking each unit and guessing context
afterwards. The implementation follows the useful RAGFlow principle that a
heading is structural context, not useful standalone knowledge:

- short heading-only units are deferred and attached to following content;
- parser-provided heading paths are preserved when available; conservative
  fallback relations recover common report subsections such as `Nguyên nhân >
  Từ nhà trường` and `Giải pháp > Đối với cá nhân`;
- the full structural path is stored as `heading_path`/`section` and prepended
  compactly to chunk text so embedding/reranking see the same context that
  provenance exposes;
- tables remain table-aware, while large prose is split to the active model's
  token budget;
- overlap prefers paragraph, line, then sentence boundaries and uses token/word
  fallback only for genuinely unstructured text; adjacent source segments keep
  explicit separators rather than producing glued strings;
- a heading with no body is not emitted as a searchable chunk merely to satisfy
  `min_chunk_tokens`. Meaningful short content at a hard boundary can still be
  retained and diagnosed.

This prevents failure modes such as a 3-token section title becoming Top-1
context, `Từ nhà trườngMôi trường...`, or a new chunk beginning in the middle
of a sentence solely because of overlap.

## Minimal Python API

This example is offline by policy: model construction fails if the named
weights are not already cached locally.

```python
from pathlib import Path

from rag import DocumentSource, RagConfig, RagModule
from rag.config import (
    EmbeddingConfig,
    PathsConfig,
    RerankerConfig,
    RetrievalConfig,
)

config = RagConfig(
    paths=PathsConfig(
        qdrant=Path(".rag_runtime/qdrant"),
        sparse=Path(".rag_runtime/sparse.db"),
        registry=Path(".rag_runtime/registry.db"),
        benchmark_output=Path(".rag_runtime/benchmarks"),
    ),
    embedding=EmbeddingConfig(
        model_name="intfloat/multilingual-e5-small",
        local_files_only=True,
        trust_remote_code=False,
    ),
    reranker=RerankerConfig(
        model_name="Alibaba-NLP/gte-multilingual-reranker-base",
        revision="8215cf04918ba6f7b6a62bb44238ce2953d8831c",
        code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4",
        local_files_only=True,
        trust_remote_code=True,
        failure_policy="fallback",
    ),
    retrieval=RetrievalConfig(top_k=5, candidate_k=50),
)

with RagModule(config) as rag:
    indexed = rag.knowledge.ingest(
        DocumentSource(
            Path("knowledge/guide.md"),
            source_type="file",
            metadata={"document_id": "guide", "department": "research"},
        ),
        operation_id="guide-ingest-2026-08-16",
    )

    result = rag.retrieve(
        "Quy trình kiểm chứng là gì?",
        mode="hybrid",
        top_k=5,
        candidate_k=50,
        rerank=True,
        include_trace=True,
    )
    for candidate in result.candidates:
        print(candidate.chunk_id, candidate.final_score, candidate.source)
```

`operation_id` is optional, but a stable caller-generated value makes a retry
of the same add, reindex, or delete operation idempotent. For raw text, use
`DocumentSource(text, source_type="text", metadata={"document_id": ...})`. A
byte source must include a basename-only `file_name` and obey the configured
size limit.

### Knowledge operations

```python
created_or_unchanged = rag.knowledge.ingest(source)
updated_or_unchanged = rag.knowledge.update(source)
new_version = rag.knowledge.reindex(source)  # force a version even if unchanged
state = rag.knowledge.inspect("guide", include_chunks=True)
deleted = rag.knowledge.delete("guide", operation_id="delete-guide")
recovered_document_ids = rag.knowledge.recover()
```

`ingest` and `update` are content-aware. `reindex` is the explicit force path.
All return typed summaries or inspectable mappings; an update never reports
success merely because one of the two indexes accepted a write.

Pass `include_trace=True` to ingest/update/reindex or retrieve when an operator
needs bounded, JSON-safe stage diagnostics. It is opt-in so normal requests do
not copy diagnostic payloads. Vector output is limited to short previews;
metadata is recursively bounded and vector/tensor-like fields are redacted.
`rag.runtime_snapshot()` returns configured and resolved parser, chunker,
model, store, retrieval, and lifecycle settings. `inspect(...,
include_chunks=True)` returns active chunk text/provenance without embeddings.

### Retrieval modes

Supported modes are `dense`, `sparse` (`bm25` is an alias), and `hybrid`.
Reranking is an independent boolean for every mode:

```python
plain = rag.retrieve("query", mode="dense", rerank=False)
lexical = rag.retrieve("query", mode="sparse", rerank=False)
fused = rag.retrieve("query", mode="hybrid", rerank=False)
reranked = rag.retrieve("query", mode="hybrid", rerank=True)
```

The legacy `mode="hybrid_rerank"` spelling is accepted only as a deprecated
compatibility alias. New code must pass `mode="hybrid", rerank=True`.

`candidate_k` is the branch/fusion pool and must be at least `top_k`; `top_k`
is the final output size. A sparse-only call does not compute a query embedding.
No mode enables the reranker implicitly.

If a requested reranker is absent, times out, or returns invalid scores, the
default `failure_policy="fallback"` returns the unchanged pre-rerank ordering
and sets `rerank_requested`, `rerank_applied`, `rerank_fallback`, and
`rerank_error` truthfully in diagnostics. Use `failure_policy="raise"` when a
request must fail instead.

### Score semantics and API contract

The ordinary public result is designed to serialize cleanly through a future
FastAPI layer. All public relevance scores are finite values in `[0, 1]`. They
are **ranking/relevance scores, not calibrated probabilities**. Backend/model
raw values remain available only in opt-in technical traces.

- `dense_score`: cosine similarity is mapped monotonically from `[-1, 1]` to
  `[0, 1]`; `dense_raw_score` keeps the backend cosine value in a trace.
- `sparse_score`: SQLite FTS5/BM25 is unbounded and corpus/query dependent, so
  its larger-is-better backend score is normalized relative to the sparse
  candidate pool for the current query. `sparse_raw_score` remains diagnostic.
- Hybrid retrieval defaults to **RRF** because dense cosine and BM25 live on
  incomparable raw scales. `fusion_score` normalizes weighted RRF by its
  theoretical per-query maximum into `[0, 1]`. Weighted fusion remains an
  explicit alternative and combines already-normalized branch relevance scores.
- `rerank_score`: probability-like reranker outputs remain bounded; scalar/logit
  outputs such as the required Alibaba GTE checkpoint are mapped monotonically
  with sigmoid. The raw model output remains `rerank_raw_score` in a trace.
- `final_score` is the canonical score for API consumers.
  `final_score_source` states whether it came from `dense`, `sparse`, `fusion`,
  or `rerank`. The older `retrieval_score` field remains for compatibility; new
  API code should prefer `final_score`.

`RetrievalResult.to_dict()` omits raw score fields by default. Calling
`retrieve(..., include_trace=True)` exposes raw stage scores inside bounded
diagnostics for debugging without changing the normal candidate contract. Do
not compare raw values across modes, queries, or model families.

`RetrievalResult` also carries query text, mode, candidates with provenance,
and query-embedding/dense/sparse/fusion/rerank/total latency fields. Metadata
filters support `=`, `!=`, `>`, `>=`, `<`, `<=`, `in`, and `contains`; both
dense and sparse paths apply the same fail-closed contract before Top-K.

## Configuration, models, and security

`rag.config.RagConfig` is the single immutable configuration root. Its nested
sections cover paths, parser/OCR limits, normalization, chunking, embedding,
dense/sparse stores, fusion, reranking, retrieval, and lifecycle. Invalid
weights, limits, templates, dimensions, or timeouts fail during construction.
`config.snapshot()` is JSON-compatible and is suitable for configuration and
verification evidence. Parser archive limits and observability bounds are part
of this same root rather than hidden parser/UI constants.

Embedding profiles distinguish document and query semantics for E5,
E5-instruct, BGE/BGE-M3, Qwen embedding, Nomic, and generic
SentenceTransformer models. Unknown/private models require explicit templates
when their semantics differ. The engine discovers or verifies vector dimension
and sequence length, preserves batch order, and reduces the batch on OOM
without skipping inputs.

After loading a model, the runtime resolves the concrete Hugging Face commit
from model/tokenizer metadata when available and always hashes the effective
input/scoring profile. Both values are exposed in `resolved_config`; embedding
revision/profile identity is also persisted with the Qdrant collection so an
incompatible checkpoint or prompt profile fails fast instead of reusing stale
vectors. A resolved commit records what was loaded—it does not replace an
operator-supplied immutable `revision` pin.

Reranker profiles cover sequence classifiers/cross-encoders (including BGE,
GTE, and Jina score conventions) and Qwen3 causal yes/no scoring. Model-family
support means that the input and score contract is implemented; validate the
exact checkpoint on a labelled dataset before production use.

Security policy is deliberate:

- Embeddings default to `trust_remote_code=False`. The required GTE reranker
  uses reviewed Hugging Face dynamic code, so its central default explicitly
  sets `trust_remote_code=True` and pins both the weight revision and the
  independent code revision shown above. Changing either is an operator-visible
  supply-chain decision.
- Set `revision` to an immutable, reviewed model revision in controlled
  deployments. The runtime records a concrete loaded commit when Hugging Face
  exposes one, but a discovered commit is evidence, not a supply-chain policy.
- Set `local_files_only=True` for offline/fail-closed operation. With `False`,
  Hugging Face libraries may download weights into their normal cache.
- File sizes, archive expansion, archive members, page counts, unit counts,
  query length, byte/upload filenames, diagnostic payloads, and extension
  mismatches are bounded or validated by configuration.
- The package emits library logs but never configures global logging handlers.
  The host application owns logging policy and secret redaction.

## Version activation and recovery

Durable ingestion requires the registry, Qdrant, and SQLite FTS5 together. The
coordinator performs this protocol:

1. Reserve a pending version and operation owner in SQLite.
2. Stage and read-back verify dense chunks.
3. Stage and read-back verify sparse chunks and cross-store identity parity.
4. Make the verified target physically ready while leaving the prior version
   searchable, then compare-and-set the registry. The registry CAS is the
   single logical visibility switch.
5. Only after CAS, mark/delete stale versions, or leave an explicit
   cleanup-pending record that restart recovery can retry.

Failed work is compensated without deleting the previous active version.
Interrupted ingest, activation, stale cleanup, and delete operations are
recovered idempotently. `create_canonical_pipeline()` runs recovery before it
returns when `lifecycle.recover_on_startup=True`, so callers cannot retrieve
through a partially initialized canonical facade.

Every retrieval captures one registry visibility epoch for all branches.
Dense and sparse search therefore cannot mix versions across a concurrent CAS;
if the epoch changes, the entire retrieval is retried a bounded number of times
with the already-computed query vector. Repeated churn fails closed.

Local embedded Qdrant and SQLite paths are designed for one service process at
a time. The registry serializes competing version ownership, but that does not
turn Qdrant local mode into a horizontally shared multi-writer service. Do not
point several application workers at the same local Qdrant directory.

## FastAPI service adapter

The production HTTP adapter lives in `rag/api/`. It is intentionally thin:
HTTP validation/serialization -> `RagModule` -> the existing two RAG flows.
It does not reimplement parsing, chunking, embedding, storage, fusion, or
reranking.

Install the API extra into the existing environment from repository root:

```powershell
rag\.venv\Scripts\python.exe -m pip install -e ".[api,test]"
```

Run one local service process:

```powershell
rag\.venv\Scripts\python.exe -m rag.api
```

Equivalent Uvicorn command:

```powershell
rag\.venv\Scripts\python.exe -m uvicorn rag.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

OpenAPI is exposed at `/docs` and `/openapi.json` by default. `RagModule` is
created once during FastAPI lifespan startup and closed during shutdown; it is
not reconstructed per request.

Important endpoints:

```text
GET    /health/live
GET    /health/ready

POST   /v1/knowledge/documents                 multipart file ingest
PUT    /v1/knowledge/documents/{document_id}   content-aware update
POST   /v1/knowledge/documents/{document_id}/reindex
GET    /v1/knowledge/documents
GET    /v1/knowledge/documents/{document_id}
GET    /v1/knowledge/documents/{document_id}/chunks
DELETE /v1/knowledge/documents/{document_id}
POST   /v1/knowledge/text
POST   /v1/knowledge/recover

POST   /v1/retrieval
```

Normal retrieval responses expose one canonical `score` in `[0, 1]` plus
provenance and context text. These are relevance scores, not calibrated
probabilities. Set `debug=true` in the JSON request only when stage scores,
raw backend scores, metadata, and bounded diagnostics are needed.

Example retrieval request:

```json
{
  "query": "Các giải pháp khắc phục tâm lý sợ sai là gì?",
  "mode": "hybrid",
  "rerank": true,
  "top_k": 5,
  "candidate_k": 50
}
```

The service uses a stable local data root. `RAG_DATA_DIR` can override it:

```powershell
$env:RAG_DATA_DIR = "A:\\NCKH\\rag-service\\.rag_runtime"
```

Without an override, a source/editable checkout resolves `.rag_runtime` from
the repository root instead of the current working directory. This prevents
Streamlit and FastAPI from silently opening different local knowledge bases.
Additional transport settings are `RAG_API_HOST`, `RAG_API_PORT`,
`RAG_API_DOCS`, and `RAG_API_LOG_LEVEL`.

Local embedded Qdrant is intended for one service process. Keep `--workers 1`
for this mode. A future multi-process deployment should move dense storage to a
Qdrant server/shared backend before increasing worker count.

## Streamlit inspection application

Install the `ui` extra and launch the application from the repository root:

```powershell
rag\.venv\Scripts\python.exe -m streamlit run rag\streamlit_app\app.py
```

The four views expose resolved configuration, all seven ingestion trace
stages, dense/sparse/fusion/rerank retrieval stages, and lifecycle/active-chunk
inspection. The UI caches one `RagModule` per configuration fingerprint and
uses only the public facade—there is no direct database, store, parser, or
pipeline orchestration in the UI. Trace collection is explicitly enabled only
for these inspection flows.

Optional environment overrides are visible and narrow:
`RAG_STREAMLIT_DATA_DIR`; `RAG_EMBEDDING_MODEL`,
`RAG_EMBEDDING_REVISION`, `RAG_EMBEDDING_DEVICE`, `RAG_EMBEDDING_DTYPE`,
`RAG_EMBEDDING_LOCAL_FILES_ONLY`; and the corresponding `RAG_RERANKER_*`
variables. Setting `RAG_RERANKER_MODEL=disabled` is the explicit way to disable
reranking. With no overrides, Streamlit uses `RagConfig` unchanged.

## Reproducible benchmarks

`rag.benchmark` supports two evidence layers:

- model evaluation with separate `embed_documents()`/`embed_query()` calls and
  a fixed reranker candidate pool;
- end-to-end comparison of sparse, dense, hybrid, and hybrid+rerank retrieval.

Use labelled relevance data. Smoke-test success is not a quality metric. The
runner functions retain per-query rankings/timings and generate JSON, CSV, and
Markdown from the same artifact:

```python
from rag.benchmark import BenchmarkCase, benchmark_modes, make_artifact, write_artifact

cases = [
    BenchmarkCase(
        query="thủ đô Việt Nam",
        relevant=frozenset({"doc-capital"}),
        case_id="q-001",
    )
]

def call(mode, *, rerank=False):
    return lambda query, **kwargs: rag.retrieve(
        query, mode=mode, rerank=rerank, **kwargs
    )

modes = benchmark_modes(
    {
        "sparse": call("sparse"),
        "dense": call("dense"),
        "hybrid": call("hybrid"),
        "hybrid_rerank": call("hybrid", rerank=True),
    },
    cases,
    top_k=5,
    candidate_k=50,
    identity_field="document_id",
)
artifact = make_artifact(
    benchmark_kind="end_to_end",
    dataset={"name": "labelled-vietnamese", "cases": len(cases)},
    modes=modes,
    config_snapshot=config.snapshot(),
    seed=42,
    model_identity={"embedding": config.embedding.model_name},
)
write_artifact(artifact, config.paths.benchmark_output / "run-001")
```

Keep the raw `.json` and `.csv` next to the generated `.md`. Do not hand-edit a
reported metric, omit failed queries, or compare runs with different corpora,
labels, models, chunking, or hardware as though they were the same experiment.

The checked-in diagnostic artifacts are historical, pre-hardening evidence for
`intfloat/multilingual-e5-small`,
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, and
`cross-encoder/ms-marco-MiniLM-L-6-v2`. They record concrete resolved commits,
profile hashes, configuration, raw rankings, and timings. The corpus is a small
authored smoke fixture, so this verifies the execution/evidence path—not broad
model quality. They do not describe the current required GTE reranker and were
not rerun as part of module completion. Benchmark execution is optional and
must be an explicit evaluation task:

```powershell
rag\.venv\Scripts\python.exe -m rag.benchmark_cli --suite all --output-dir rag\benchmark_results
```

Use `--allow-download` only when downloading model weights is explicitly
authorized.

## Verification commands

Run every command from the repository root with the existing virtual
environment:

```powershell
rag\.venv\Scripts\python.exe -m rag.verification
rag\.venv\Scripts\python.exe -m pytest rag\tests\unit -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\parser -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\integration rag\tests\recovery -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\model -q
rag\.venv\Scripts\python.exe -m pytest rag\tests\e2e\test_real_pipeline.py -q
rag\.venv\Scripts\python.exe -m pytest -q
```

The model capability tests mostly use controlled models to prove formatting,
score extraction, order, dimension, revision/profile identity, and OOM
behavior. The real vertical E2E uses the public `RagModule`, cached-only
`intfloat/multilingual-e5-small`, the pinned required Alibaba GTE reranker, a
real Markdown parser, Qdrant, SQLite FTS5, every retrieval mode, inspect,
update, recovery, and delete. This proves executable integration, not ranking
quality. The Tesseract parser tests need the host executable and
language data. Treat an unavailable external prerequisite as unverified, not
as evidence of model quality.

See [RAGFLOW_REFERENCE_AUDIT.md](RAGFLOW_REFERENCE_AUDIT.md) for the reference
matrix, [AUDIT_REPORT.md](AUDIT_REPORT.md) for the repository audit, and
[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for benchmark evidence and artifact
provenance.

## Known limits

- Real-world PDF layout diversity, rotated/irregular tables, handwriting, and
  OCR language breadth are not equivalent to RAGFlow's DeepDOC vision stack.
- Auto OCR fallback ordering and aggregated failure diagnostics are tested, but
  real RapidOCR model/runtime breadth still needs environment-specific
  validation; Tesseract quality depends on host binaries and language packs.
- Real cached CPU completion evidence covers the default E5 embedding and the
  exact pinned Alibaba GTE reranker. Other BGE/Qwen/Jina/Nomic/private
  checkpoints and the CUDA execution path are not real-model verified.
- Streamlit's public-facade AppTest and a headless HTTP health probe are
  verified; interactive browser rendering across every host/browser remains an
  environment-specific check.
- Local stores are not a distributed transaction or multi-node deployment.
- Retrieval quality remains dataset-specific. No generation faithfulness or
  answer-quality claim is in scope because this package returns chunks only.
