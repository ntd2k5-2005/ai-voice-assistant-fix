"""Deterministic model checkpoint and prompt-profile identity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def resolved_model_revision(
    model: Any,
    *,
    tokenizer: Any | None = None,
    requested: str | None = None,
) -> str | None:
    """Prefer the concrete Hub commit loaded by Transformers when available."""

    candidates = [
        getattr(getattr(model, "config", None), "_commit_hash", None),
        getattr(model, "_commit_hash", None),
        (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
        if tokenizer is not None
        else None,
    ]
    try:
        first = model[0]
    except (TypeError, KeyError, IndexError, AttributeError):
        first = None
    if first is not None:
        candidates.append(
            getattr(
                getattr(getattr(first, "auto_model", None), "config", None),
                "_commit_hash",
                None,
            )
        )
    candidates.append(requested)
    for value in candidates:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def profile_fingerprint(profile: Any) -> str:
    """Hash all declared profile fields that can alter model input/scoring."""

    values = {
        key: value
        for key, value in vars(profile).items()
        if isinstance(value, (str, bool, int, float)) or value is None
    }
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["profile_fingerprint", "resolved_model_revision"]
