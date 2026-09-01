from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def model_device_dtype(model) -> tuple[torch.device, torch.dtype]:
    """Use the loaded model as the runtime source of truth."""
    param = next(model.parameters())
    return param.device, param.dtype


def normalize_model_inputs(
    inputs: Mapping[str, Any],
    model,
) -> dict[str, Any]:
    """
    Move processor outputs to the model device.

    Floating tensors (e.g. Whisper input_features) also adopt the model
    dtype. Integer/bool tensors keep their dtype so masks/token IDs are not
    accidentally converted to float16.
    """
    device, dtype = model_device_dtype(model)
    normalized: dict[str, Any] = {}

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                normalized[key] = value.to(device=device, dtype=dtype)
            else:
                normalized[key] = value.to(device=device)
        else:
            normalized[key] = value
    return normalized


def safe_max_new_tokens(
    model,
    requested: int,
    *,
    reserve_tokens: int = 8,
    default_target_positions: int = 448,
) -> int:
    """
    Reserve room for Whisper decoder prefix/language/task tokens.

    This fixes the integration failure where max_new_tokens=448 plus
    decoder prompt tokens exceeded max_target_positions=448.
    """
    if requested < 1:
        raise ValueError("requested must be >= 1")
    max_positions = int(
        getattr(model.config, "max_target_positions", default_target_positions)
    )
    return min(requested, max(1, max_positions - reserve_tokens))
