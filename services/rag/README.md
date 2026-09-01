# Robust RAG service

The deployable product is the standalone `rag` package. See
[`rag/README.md`](rag/README.md) for architecture, supported inputs, public API,
Streamlit launch instructions, security policy, verification, and known limits.

`ragflow-main/` is reference material only and is never a runtime dependency.


## Service entry points

- Python facade: `rag.RagModule`
- FastAPI service: `python -m rag.api` (`/docs`, `/v1/retrieval`, `/v1/knowledge/...`)
- Streamlit inspection UI: `python -m streamlit run rag/streamlit_app/app.py`

Use `RAG_DATA_DIR` to pin one persistent local knowledge directory for every entry point.
