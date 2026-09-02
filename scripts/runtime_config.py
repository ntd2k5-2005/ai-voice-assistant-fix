from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


REQUIRED_COMPONENTS = {
    "stt", "vad", "embedding", "sparse", "reranker", "llm", "tts"
}
PLACED_COMPONENTS = {"stt", "embedding", "reranker", "llm", "tts"}
VALID_DEVICES = {
    "stt": {"cpu", "cuda"},
    "embedding": {"cpu", "cuda", "mps"},
    "reranker": {"cpu", "cuda", "mps"},
    "llm": {"cpu", "adaptive-offload"},
    "tts": {"cpu"},
}
VALID_DTYPES = {"float32", "float16", "bfloat16"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"runtime configuration file does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"runtime configuration root must be an object: {path}")
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {path}")
    return value


def load_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "config" / "model_manifest.json")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != REQUIRED_COMPONENTS:
        raise ValueError("model manifest must declare exactly the baseline components")
    for name, component in components.items():
        if not isinstance(component, dict) or not str(component.get("runtime") or "").strip():
            raise ValueError(f"manifest component {name!r} lacks a runtime")
        if name != "sparse" and not str(component.get("model_id") or "").strip():
            raise ValueError(f"manifest component {name!r} lacks model_id")
    return manifest


def load_profile(root: Path, name: str | None = None) -> tuple[str, dict[str, Any]]:
    document = _read_json(root / "config" / "runtime_profiles.json")
    profile_name = (name or os.getenv("RUNTIME_PROFILE") or document.get("default_profile") or "").strip()
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        choices = ", ".join(sorted(profiles or {}))
        raise ValueError(f"unknown runtime profile {profile_name!r}; choose one of: {choices}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"runtime profile {profile_name!r} must be an object")
    placement = profile.get("placement")
    if not isinstance(placement, dict) or set(placement) != PLACED_COMPONENTS:
        raise ValueError(f"runtime profile {profile_name!r} has incomplete placement")
    for component, allowed in VALID_DEVICES.items():
        configured = placement[component]
        if not isinstance(configured, dict) or configured.get("device") not in allowed:
            raise ValueError(f"runtime profile {profile_name!r} has invalid {component} device")
        if component in {"stt", "embedding", "reranker"}:
            dtype = configured.get("dtype")
            if dtype not in VALID_DTYPES:
                raise ValueError(f"runtime profile {profile_name!r} has invalid {component} dtype")
            if configured["device"] == "cpu" and dtype != "float32":
                raise ValueError(f"runtime profile {profile_name!r} must use float32 for {component} on CPU")
    llm = placement["llm"]
    if not isinstance(llm.get("num_gpu"), int) or llm["num_gpu"] < -1:
        raise ValueError(f"runtime profile {profile_name!r} has invalid llm.num_gpu")
    if (llm["device"] == "cpu") != (llm["num_gpu"] == 0):
        raise ValueError(f"runtime profile {profile_name!r} has inconsistent LLM placement")
    ports = profile.get("ports")
    if not isinstance(ports, dict) or set(ports) != {"stt", "rag", "llm", "tts", "web"}:
        raise ValueError(f"runtime profile {profile_name!r} has incomplete ports")
    if any(not isinstance(value, int) or not 1 <= value <= 65535 for value in ports.values()):
        raise ValueError(f"runtime profile {profile_name!r} contains an invalid port")
    if len(set(ports.values())) != len(ports):
        raise ValueError(f"runtime profile {profile_name!r} reuses a port")
    return profile_name, profile


def service_environments(
    root: Path,
    manifest: Mapping[str, Any],
    profile_name: str,
    profile: Mapping[str, Any],
    base: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    components = manifest["components"]
    placement = profile["placement"]
    ports = profile["ports"]
    inherited = dict(os.environ if base is None else base)
    offline = inherited.get("RUNTIME_OFFLINE", "false").strip().casefold() in {
        "1", "true", "yes", "on"
    }
    cache_root = root / ".model_cache" / "huggingface"
    rag_data = root / ".runtime_data" / profile_name / "rag"
    common = {
        "RUNTIME_PROFILE": profile_name,
        "MODEL_MANIFEST_ID": str(manifest["manifest_id"]),
        "MODEL_MANIFEST_PATH": str(root / "config" / "model_manifest.json"),
        "HF_HOME": str(cache_root),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    if offline:
        common |= {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    def merged(values: Mapping[str, Any]) -> dict[str, str]:
        return inherited | common | {key: str(value) for key, value in values.items() if value is not None}

    stt = components["stt"]
    embedding = components["embedding"]
    reranker = components["reranker"]
    llm = components["llm"]
    tts = components["tts"]
    environments = {
        "stt": merged({
            "MODEL_ID": stt["model_id"],
            "MODEL_REVISION": stt.get("revision"),
            "MODEL_DEVICE": placement["stt"]["device"],
            "MODEL_DTYPE": placement["stt"]["dtype"],
            "MODEL_LOCAL_FILES_ONLY": str(offline).lower(),
        }),
        "rag": merged({
            "RAG_API_HOST": "127.0.0.1",
            "RAG_API_PORT": ports["rag"],
            "RAG_DATA_DIR": rag_data,
            "RAG_EMBEDDING_MODEL": embedding["model_id"],
            "RAG_EMBEDDING_REVISION": embedding.get("revision"),
            "RAG_EMBEDDING_DEVICE": placement["embedding"]["device"],
            "RAG_EMBEDDING_DTYPE": placement["embedding"]["dtype"],
            "RAG_EMBEDDING_LOCAL_FILES_ONLY": str(offline).lower(),
            "RAG_RERANKER_MODEL": reranker["model_id"],
            "RAG_RERANKER_REVISION": reranker.get("revision"),
            "RAG_RERANKER_CODE_REVISION": reranker.get("code_revision"),
            "RAG_RERANKER_DEVICE": placement["reranker"]["device"],
            "RAG_RERANKER_DTYPE": placement["reranker"]["dtype"],
            "RAG_RERANKER_LOCAL_FILES_ONLY": str(offline).lower(),
        }),
        "llm": merged({
            "API_HOST": "127.0.0.1",
            "API_PORT": ports["llm"],
            "OLLAMA_URL": inherited.get("OLLAMA_URL", "http://127.0.0.1:11434"),
            "MODEL_NAME": llm["model_id"],
            "OLLAMA_PLACEMENT": placement["llm"]["device"],
            "OLLAMA_NUM_GPU": placement["llm"]["num_gpu"],
        }),
        "tts": merged({
            "TTS_PROFILE": "realtime",
            "TTS_VIENEU_MODE": tts["model_id"].removeprefix("vieneu-"),
            "TTS_BACKBONE_MODEL_ID": tts.get("backbone_model_id"),
            "TTS_BACKBONE_REVISION": tts.get("backbone_revision"),
            "TTS_CODEC_MODEL_ID": tts.get("codec_model_id"),
            "TTS_CODEC_REVISION": tts.get("codec_revision"),
            "VIENEU_UPSTREAM_COMMIT": tts["upstream_commit"],
            "VIENEU_UPSTREAM_REPOSITORY": tts["upstream_repository"],
        }),
    }
    environments["web"] = merged({
        "STT_BASE_URL": f"http://127.0.0.1:{ports['stt']}",
        "RAG_BASE_URL": f"http://127.0.0.1:{ports['rag']}",
        "LLM_BASE_URL": f"http://127.0.0.1:{ports['llm']}",
        "TTS_BASE_URL": f"http://127.0.0.1:{ports['tts']}",
        "DEMO_HOST": "127.0.0.1",
        "DEMO_PORT": ports["web"],
    })
    return environments


__all__ = ["load_manifest", "load_profile", "service_environments"]
