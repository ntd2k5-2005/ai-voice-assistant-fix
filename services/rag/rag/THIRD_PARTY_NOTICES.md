# Third-party reference notice

This project was developed with reference to the RAGFlow source tree included
next to this package as `ragflow-main/`. RAGFlow is Copyright The InfiniFlow
Authors and is licensed under the Apache License, Version 2.0. The complete
license is available at `ragflow-main/LICENSE` in this workspace and from the
Apache Software Foundation.

The production package does not import or execute RAGFlow. Its implementations
are isolated reimplementations shaped by these observable design patterns:

- parser-specific structured units, page/row/slide provenance, per-page PDF
  fallback, and explicit table handling from `deepdoc/parser/`;
- token-budget and structure-aware merge behavior from `rag/app/naive.py`,
  `rag/nlp/__init__.py`, and `rag/flow/chunker/`;
- distinct query/document embedding calls and order-preserving batches from
  `rag/llm/embedding_model.py`;
- active-availability filtering, bounded candidate pools, score diagnostics,
  and normalized reranking from `rag/nlp/search.py` and
  `rag/llm/rerank_model.py`.

No RAGFlow models, datasets, services, or runtime modules are redistributed as
part of the `rag` Python package.
