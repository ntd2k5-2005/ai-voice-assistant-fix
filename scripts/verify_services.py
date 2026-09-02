from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from runtime_config import load_manifest, load_profile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()
    if not args.profile:
        parser.error("--profile is required")
    manifest = load_manifest(ROOT)
    profile_name, profile = load_profile(ROOT, args.profile)
    ports = profile["ports"]
    endpoints = {
        name: f"http://127.0.0.1:{ports[name]}/health/ready"
        for name in ("stt", "rag", "llm", "tts")
    }
    failed = False
    payloads = {}
    for name, url in endpoints.items():
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
                payloads[name] = payload
                print(f"{name.upper():4} HTTP {response.status}: {payload}")
                failed |= not (200 <= response.status < 300)
        except Exception as exc:
            failed = True
            print(f"{name.upper():4} FAIL: {exc}")

    tts = payloads.get("tts") or {}
    realtime_ok = (
        tts.get("profile") == "realtime"
        and tts.get("streaming_mode") == "frame-pcm"
        and "onnx" in str(tts.get("backend") or "").lower()
    )
    if not realtime_ok:
        failed = True
        print("TTS V10 FAIL: expected profile=realtime, backend=onnx, streaming_mode=frame-pcm")
    else:
        print("TTS V10 realtime contract PASS")

    expected_manifest = manifest["manifest_id"]
    for name, payload in payloads.items():
        if payload.get("model_manifest_id") != expected_manifest:
            failed = True
            print(
                f"{name.upper()} MANIFEST FAIL: expected={expected_manifest} "
                f"actual={payload.get('model_manifest_id')}"
            )

    placement = profile["placement"]
    stt = payloads.get("stt") or {}
    if stt and stt.get("device") != placement["stt"]["device"]:
        failed = True
        print("STT placement does not match the selected profile")
    rag_models = (payloads.get("rag") or {}).get("models") or {}
    for component in ("embedding", "reranker"):
        actual = (rag_models.get(component) or {}).get("device")
        if actual and actual != placement[component]["device"]:
            failed = True
            print(f"RAG {component} placement does not match the selected profile")
    llm = payloads.get("llm") or {}
    if llm and llm.get("configured_num_gpu") != placement["llm"]["num_gpu"]:
        failed = True
        print("LLM configured num_gpu does not match the selected profile")
    expected_digest = manifest["components"]["llm"].get("manifest_digest")
    if expected_digest and llm.get("model_digest") != expected_digest:
        failed = True
        print("LLM digest does not match the model manifest")
    tts_manifest = manifest["components"]["tts"]
    tts_artifacts = (payloads.get("tts") or {}).get("model_artifacts") or {}
    for name in ("backbone", "codec"):
        expected = tts_manifest.get(f"{name}_revision")
        artifact = tts_artifacts.get(name) or {}
        actual = artifact.get("resolved_revision")
        if expected and actual != expected:
            failed = True
            print(f"TTS {name} revision does not match the model manifest")
        policy = artifact.get("revision_policy") or {}
        if expected and policy.get("status") != "enforced":
            failed = True
            print(
                f"TTS {name} revision is not enforced: "
                f"mechanism={policy.get('mechanism')} limitation={policy.get('limitation')}"
            )
        elif expected:
            print(
                f"TTS {name} revision enforcement PASS: "
                f"{policy.get('mechanism')}"
            )

    print(f"PROFILE {profile_name} verification {'FAIL' if failed else 'PASS'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
