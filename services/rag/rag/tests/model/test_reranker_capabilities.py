from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rag.contracts import RetrievalCandidate
from rag.retrieval.reranker_capabilities import (
    RerankerProfileOverrides,
    resolve_reranker_profile,
)
from rag.retrieval.reranking import Reranker, RerankerOOMError


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch(
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in self.items()
            }
        )


class FakeTokenizer:
    model_max_length = 64
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, pairs, **kwargs):
        self.calls.append((list(pairs), dict(kwargs)))
        rows = []
        for pair in pairs:
            text = pair[1] if isinstance(pair, tuple) else pair
            if not isinstance(pair, tuple) and "<Document>:" in text:
                text = text.rsplit("<Document>:", 1)[-1]
            rows.append([2 if "relevant" in text else 0, 1])
        return FakeBatch(
            input_ids=torch.tensor(rows, dtype=torch.long),
            attention_mask=torch.ones((len(rows), 2), dtype=torch.long),
        )

    def encode(self, value, add_special_tokens=False):
        return [10 if value == "yes" else 11]

    def pad(self, values, padding=True, return_tensors="pt"):
        rows = values["input_ids"]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + list(row) for row in rows]
        attention = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return FakeBatch(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(attention, dtype=torch.long),
        )


class FakeClassifier:
    def __init__(self, *, width=2, oom_above=None, invalid=False):
        self.width = width
        self.oom_above = oom_above
        self.invalid = invalid
        self.calls = []
        self.config = SimpleNamespace(
            max_position_embeddings=128,
            id2label={0: "irrelevant", 1: "relevant"},
        )

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        self.calls.append(input_ids.shape[0])
        if self.oom_above is not None and input_ids.shape[0] > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        if self.invalid:
            return SimpleNamespace(logits=torch.zeros((input_ids.shape[0], 2, 2)))
        strength = input_ids[:, 0].float()
        if self.width == 1:
            logits = strength.reshape(-1, 1)
        else:
            logits = torch.stack([torch.zeros_like(strength), strength], dim=1)
        return SimpleNamespace(logits=logits)


def make_reranker(model_name="cross-encoder/test", **kwargs):
    tokenizer = kwargs.pop("tokenizer", FakeTokenizer())
    model = kwargs.pop("model", FakeClassifier())
    return Reranker(
        model_name,
        device="cpu",
        tokenizer=tokenizer,
        model=model,
        batch_size=kwargs.pop("batch_size", 4),
        **kwargs,
    )


def candidates():
    return [
        RetrievalCandidate(chunk_id="z", text="unrelated passage"),
        RetrievalCandidate(chunk_id="a", text="relevant passage"),
    ]


def test_family_resolver_selects_architecture_and_score_contract():
    assert resolve_reranker_profile("BAAI/bge-reranker-v2-m3").score_extraction == "sigmoid"
    gte = resolve_reranker_profile("Alibaba-NLP/gte-multilingual-reranker-base")
    assert gte.family == "gte-reranker"
    assert gte.requires_remote_code is True
    assert gte.score_extraction == "scalar"
    assert gte.max_length == 8192
    assert gte.padding_side == "right"
    assert gte.position_ids_buffer_path == "new.embeddings.position_ids"
    other_gte = resolve_reranker_profile("custom/gte-reranker-experimental")
    assert other_gte.family == "gte-reranker"
    assert other_gte.position_ids_buffer_path == ""
    qwen = resolve_reranker_profile("Qwen/Qwen3-Reranker-0.6B")
    assert qwen.architecture == "causal_yes_no"
    assert "<Query>: q" in qwen.format_pair("q", "d")


def test_sequence_classifier_extracts_positive_probability_and_ranks_stably():
    reranker = make_reranker()
    ranked = reranker.rerank_candidates(candidates(), "question", top_k=2)
    assert [candidate.chunk_id for candidate in ranked] == ["a", "z"]
    assert ranked[0].rerank_score > ranked[1].rerank_score
    assert reranker.resolved_config["architecture"] == "sequence_classification"
    assert reranker.max_length == 64
    assert len(reranker.resolved_config["profile_hash"]) == 64


def test_concrete_reranker_checkpoint_revision_is_exposed():
    model = FakeClassifier()
    model.config._commit_hash = "reranker-commit"
    reranker = make_reranker(model=model)
    assert reranker.resolved_config["revision"] == "reranker-commit"


def test_scalar_bge_scores_use_profile_specific_sigmoid():
    reranker = make_reranker(
        "BAAI/bge-reranker-v2-m3", model=FakeClassifier(width=1)
    )
    ranked = reranker.rerank_candidates(candidates(), "question", top_k=2)
    assert 0 < ranked[1].rerank_score < ranked[0].rerank_score < 1


def test_gte_uses_raw_scalar_logits_and_restores_position_ids():
    class GTEEmbeddings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(8, 2)
            self.register_buffer(
                "position_ids",
                torch.full((128,), 99, dtype=torch.long),
                persistent=False,
            )

    class GTEBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = GTEEmbeddings()

    class GTEClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.new = GTEBackbone()
            self.config = SimpleNamespace(
                max_position_embeddings=128,
                id2label={0: "LABEL_0"},
            )

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=input_ids[:, :1].float())

    tokenizer = FakeTokenizer()
    reranker = make_reranker(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        tokenizer=tokenizer,
        model=GTEClassifier(),
        trust_remote_code=True,
    )
    ranked = reranker.rerank_candidates(candidates(), "question", top_k=2)
    assert ranked[0].rerank_score == pytest.approx(2.0)
    assert ranked[1].rerank_score == pytest.approx(0.0)
    assert tokenizer.padding_side == "right"
    assert reranker.max_length == 64
    assert torch.equal(
        reranker.model.new.embeddings.position_ids,
        torch.arange(128),
    )
    assert reranker.resolved_config["compatibility_repairs"] == (
        "new.embeddings.position_ids",
    )
    assert reranker.resolved_config["dtype"] == "float32"


def test_explicit_positive_label_index_supports_nonstandard_classifier():
    class ReversedClassifier(FakeClassifier):
        def __call__(self, input_ids, **kwargs):
            strength = input_ids[:, 0].float()
            return SimpleNamespace(logits=torch.stack([strength, torch.zeros_like(strength)], dim=1))

    reranker = make_reranker(
        model=ReversedClassifier(),
        profile_overrides=RerankerProfileOverrides(
            score_extraction="positive_probability", positive_label_index=0
        ),
    )
    assert reranker.rerank_candidates(candidates(), "q", top_k=1)[0].chunk_id == "a"


def test_oom_reduces_batch_and_preserves_candidate_score_mapping():
    model = FakeClassifier(oom_above=2)
    reranker = make_reranker(model=model, batch_size=4)
    values = [
        RetrievalCandidate(
            chunk_id=str(index),
            text="relevant passage" if index % 2 else "unrelated passage",
        )
        for index in range(6)
    ]
    ranked = reranker.rerank("q", values, top_k=6)
    assert model.calls == [4, 2, 2, 2]
    assert len(ranked) == 6
    assert reranker.current_batch_size == 2


def test_oom_at_batch_one_and_invalid_logits_are_explicit():
    with pytest.raises(RerankerOOMError, match="retry policy exhausted"):
        make_reranker(model=FakeClassifier(oom_above=0), batch_size=1).rerank(
            "q", candidates(), top_k=1
        )
    with pytest.raises(RuntimeError, match="unsupported reranker logits shape"):
        make_reranker(model=FakeClassifier(invalid=True)).rerank("q", candidates(), top_k=1)


def test_empty_candidates_validation_and_deterministic_ties():
    reranker = make_reranker(model=FakeClassifier(width=1))
    assert reranker.rerank("q", [], top_k=1) == []
    tied = [
        RetrievalCandidate(chunk_id="b", text="none"),
        RetrievalCandidate(chunk_id="a", text="none"),
    ]
    assert [item.chunk_id for item, _ in reranker.rerank("q", tied, top_k=2)] == ["a", "b"]
    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank("   ", candidates(), top_k=1)
    with pytest.raises(ValueError, match="non-empty text"):
        reranker.rerank("q", [RetrievalCandidate(chunk_id="x")], top_k=1)


def test_remote_code_requirement_is_never_implicitly_enabled():
    with pytest.raises(ValueError, match="explicit trust_remote_code"):
        make_reranker(
            profile_overrides=RerankerProfileOverrides(requires_remote_code=True)
        )
    reranker = make_reranker(
        profile_overrides=RerankerProfileOverrides(requires_remote_code=True),
        trust_remote_code=True,
    )
    assert reranker.resolved_config["trust_remote_code"] is True


def test_qwen_causal_profile_uses_yes_no_probability_adapter():
    class CausalModel(FakeClassifier):
        def __call__(self, input_ids, **kwargs):
            batch, sequence = input_ids.shape
            logits = torch.zeros((batch, sequence, 20), dtype=torch.float32)
            # Relevant prompts receive a stronger 'yes' logit.
            logits[:, -1, 10] = (input_ids == 2).any(dim=1).float() * 2
            logits[:, -1, 11] = 0
            return SimpleNamespace(logits=logits)

    reranker = make_reranker(
        "Qwen/Qwen3-Reranker-0.6B", model=CausalModel(), tokenizer=FakeTokenizer()
    )
    ranked = reranker.rerank_candidates(candidates(), "question", top_k=2)
    assert [candidate.chunk_id for candidate in ranked] == ["a", "z"]


def test_close_prevents_reuse():
    reranker = make_reranker()
    reranker.close()
    with pytest.raises(RuntimeError, match="closed"):
        reranker.rerank("q", candidates(), top_k=1)
