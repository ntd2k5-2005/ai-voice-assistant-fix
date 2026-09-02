from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from runtime_config import load_manifest, load_profile, service_environments  # noqa: E402
from validate_acceptance import (  # noqa: E402
    JSONL_REQUIREMENTS,
    STT_GROUPS,
    validate_jsonl,
    validate_stt_group,
)


def test_manifest_preserves_the_current_source_baseline():
    manifest = load_manifest(ROOT)
    components = manifest["components"]
    assert components["stt"]["model_id"] == "vinai/PhoWhisper-base"
    assert components["stt"]["revision"] == "7ebdb9e88f5cc5271fb88f4d642c82ff9388650e"
    assert components["embedding"]["model_id"] == "intfloat/multilingual-e5-small"
    assert components["embedding"]["revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert components["reranker"]["revision"] == "8215cf04918ba6f7b6a62bb44238ce2953d8831c"
    assert components["llm"]["model_id"] == "qwen3.5:4b"
    assert components["llm"]["manifest_digest"] == (
        "2a654d98e6fba55d452b7043684e9b57"
        "a947e393bbffa62485a7aac05ee4eefd"
    )

    tts_runtime = json.loads(
        (ROOT / "services" / "tts" / "runtime.json").read_text(encoding="utf-8")
    )
    assert components["tts"]["upstream_commit"] == tts_runtime["commit"]
    assert components["tts"]["upstream_repository"] == tts_runtime["repository"]
    assert components["tts"]["backbone_revision"] == "2da0efab622a1722125991736524f080b751ef5b"
    assert components["tts"]["codec_revision"] == "ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae"


def test_every_runtime_profile_has_explicit_non_auto_placement():
    document = json.loads(
        (ROOT / "config" / "runtime_profiles.json").read_text(encoding="utf-8")
    )
    for name in document["profiles"]:
        resolved_name, profile = load_profile(ROOT, name)
        assert resolved_name == name
        assert all(value["device"] != "auto" for value in profile["placement"].values())


def test_local_balanced_profile_builds_isolated_service_environments():
    manifest = load_manifest(ROOT)
    name, profile = load_profile(ROOT, "local-balanced")
    values = service_environments(ROOT, manifest, name, profile, {})

    assert values["stt"]["MODEL_DEVICE"] == "cpu"
    assert values["rag"]["RAG_EMBEDDING_DEVICE"] == "cpu"
    assert values["rag"]["RAG_RERANKER_DEVICE"] == "cpu"
    assert values["llm"]["OLLAMA_NUM_GPU"] == "-1"
    assert values["tts"]["TTS_VIENEU_MODE"] == "v3turbo"
    assert values["web"]["STT_BASE_URL"] == "http://127.0.0.1:8100"
    assert values["web"]["RUNTIME_PROFILE"] == "local-balanced"
    assert values["web"]["MODEL_MANIFEST_ID"] == manifest["manifest_id"]
    assert all(value["HF_HUB_DISABLE_TELEMETRY"] == "1" for value in values.values())


def test_offline_runtime_forces_huggingface_components_to_cache_only():
    manifest = load_manifest(ROOT)
    name, profile = load_profile(ROOT, "local-cpu")
    values = service_environments(
        ROOT, manifest, name, profile, {"RUNTIME_OFFLINE": "true"}
    )
    assert values["stt"]["MODEL_LOCAL_FILES_ONLY"] == "true"
    assert values["rag"]["RAG_EMBEDDING_LOCAL_FILES_ONLY"] == "true"
    assert values["rag"]["RAG_RERANKER_LOCAL_FILES_ONLY"] == "true"
    assert all(value["HF_HUB_OFFLINE"] == "1" for value in values.values())


def test_empty_acceptance_files_are_valid_but_have_zero_cases():
    assert all(validate_jsonl(path, required) == 0 for path, required in JSONL_REQUIREMENTS.items())
    assert all(
        validate_stt_group(ROOT / "acceptance" / "stt" / group) == 0
        for group in STT_GROUPS
    )
    assert json.loads(
        (ROOT / "acceptance" / "realtime" / "scenarios.json").read_text(encoding="utf-8")
    ) == []


def test_services_expose_manifest_driven_runtime_identity():
    stt = (ROOT / "services" / "stt" / "app.py").read_text(encoding="utf-8")
    rag = (ROOT / "services" / "rag" / "rag" / "api" / "app.py").read_text(encoding="utf-8")
    llm = (ROOT / "services" / "llm" / "llm" / "service.py").read_text(encoding="utf-8")
    tts = (ROOT / "services" / "tts" / "api.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "start_services.py").read_text(encoding="utf-8")

    assert 'os.getenv("MODEL_DEVICE", "auto")' in stt
    assert "config_from_environment()" in rag
    assert 'options["num_gpu"]' in llm
    assert '"model_manifest_id"' in tts
    assert "service_environments" in launcher
