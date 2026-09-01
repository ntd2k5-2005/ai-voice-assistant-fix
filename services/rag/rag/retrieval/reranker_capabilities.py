"""Central model-family profiles for local rerankers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


RerankerArchitecture = Literal["sequence_classification", "causal_yes_no"]
PaddingSide = Literal["left", "right"]
ScoreExtraction = Literal[
    "auto",
    "scalar",
    "sigmoid",
    "positive_logit",
    "positive_probability",
    "yes_no_probability",
]


@dataclass(frozen=True)
class RerankerProfile:
    family: str
    architecture: RerankerArchitecture = "sequence_classification"
    score_extraction: ScoreExtraction = "auto"
    positive_label_index: int | None = None
    higher_is_better: bool = True
    pair_template: str = "{query}\n{document}"
    instruction: str = ""
    prefix: str = ""
    suffix: str = ""
    requires_remote_code: bool = False
    max_length: int | None = None
    padding_side: PaddingSide | None = None
    position_ids_buffer_path: str = ""

    def __post_init__(self) -> None:
        if "{query}" not in self.pair_template or "{document}" not in self.pair_template:
            raise ValueError("reranker pair_template must contain {query} and {document}")
        if self.max_length is not None and self.max_length <= 1:
            raise ValueError("reranker profile max_length must be greater than one")

    def format_pair(self, query: str, document: str) -> str:
        return self.pair_template.format(
            query=query,
            document=document,
            instruction=self.instruction,
        )


@dataclass(frozen=True)
class RerankerProfileOverrides:
    family: str | None = None
    architecture: RerankerArchitecture | None = None
    score_extraction: ScoreExtraction | None = None
    positive_label_index: int | None = None
    higher_is_better: bool | None = None
    pair_template: str | None = None
    instruction: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    requires_remote_code: bool | None = None
    max_length: int | None = None
    padding_side: PaddingSide | None = None
    position_ids_buffer_path: str | None = None


_GENERIC = RerankerProfile(family="cross-encoder")
_BGE = RerankerProfile(family="bge-reranker", score_extraction="sigmoid")
_GTE = RerankerProfile(family="gte-reranker")
_GTE_MULTILINGUAL = RerankerProfile(
    family="gte-reranker",
    score_extraction="scalar",
    requires_remote_code=True,
    max_length=8192,
    padding_side="right",
    # Alibaba-NLP/new-impl registers this non-persistent buffer during model
    # construction. Transformers 5's meta-device loading can leave it
    # uninitialized, so the adapter restores the declared arange buffer.
    position_ids_buffer_path="new.embeddings.position_ids",
)
_JINA = RerankerProfile(family="jina-reranker", score_extraction="sigmoid")
_QWEN3 = RerankerProfile(
    family="qwen3-reranker",
    architecture="causal_yes_no",
    score_extraction="yes_no_probability",
    pair_template="<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
    instruction=(
        "Given a web search query, retrieve relevant passages that answer the query"
    ),
    prefix=(
        '<|im_start|>system\nJudge whether the Document meets the requirements based on '
        'the Query and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n'
    ),
    suffix="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    padding_side="left",
)


def resolve_reranker_profile(
    model_name: str,
    overrides: RerankerProfileOverrides | None = None,
) -> RerankerProfile:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("reranker model_name must be a non-empty string")
    name = model_name.casefold()
    if "qwen3" in name and "rerank" in name:
        profile = _QWEN3
    elif "bge" in name and "rerank" in name:
        profile = _BGE
    elif "gte-multilingual-reranker-base" in name:
        profile = _GTE_MULTILINGUAL
    elif "gte" in name and "rerank" in name:
        profile = _GTE
    elif "jina" in name and "rerank" in name:
        profile = _JINA
    else:
        profile = _GENERIC
    if overrides is None:
        return profile
    changes = {key: value for key, value in vars(overrides).items() if value is not None}
    return replace(profile, **changes)


__all__ = [
    "RerankerArchitecture",
    "PaddingSide",
    "ScoreExtraction",
    "RerankerProfile",
    "RerankerProfileOverrides",
    "resolve_reranker_profile",
]
