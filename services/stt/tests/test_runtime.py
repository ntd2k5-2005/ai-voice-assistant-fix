from types import SimpleNamespace

import torch

from runtime import normalize_model_inputs, safe_max_new_tokens


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
        self.config = SimpleNamespace(max_target_positions=448)


def test_normalize_model_inputs_casts_only_floating_tensors():
    model = DummyModel()
    features = torch.ones(2, 3, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.int64)

    out = normalize_model_inputs(
        {"input_features": features, "attention_mask": mask, "meta": "x"},
        model,
    )

    assert out["input_features"].dtype == torch.float16
    assert out["attention_mask"].dtype == torch.int64
    assert out["meta"] == "x"


def test_safe_max_new_tokens_reserves_decoder_prefix_space():
    model = DummyModel()
    assert safe_max_new_tokens(model, 448, reserve_tokens=8) == 440
    assert safe_max_new_tokens(model, 400, reserve_tokens=8) == 400
