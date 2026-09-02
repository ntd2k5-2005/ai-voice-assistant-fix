from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from runtime_config import load_manifest, load_profile, service_environments


ROOT = Path(__file__).resolve().parents[1]
BACKBONE_FILES = (
    "onnx_int8/vieneu_prefill.onnx",
    "onnx_int8/vieneu_decode_step.onnx",
    "onnx_int8/vieneu_acoustic_cached.onnx",
    "onnx_int8/vieneu_backbone_shared.data",
    "onnx_int8/vieneu_v3_heads.npz",
    "onnx_int8/config.json",
    "onnx_int8/tokenizer.json",
)
CODEC_FILES = (
    "moss_audio_tokenizer_decode_full.onnx",
    "moss_audio_tokenizer_decode_shared.data",
    "moss_audio_tokenizer_decode_step.onnx",
    "codec_browser_onnx_meta.json",
    "moss_audio_tokenizer_encode.onnx",
    "moss_audio_tokenizer_encode.data",
)


def _verify_snapshot(
    path: Path,
    revision: str,
    *,
    required_files: tuple[str, ...],
) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"downloaded Hugging Face snapshot is missing: {resolved}")
    if resolved.name != revision:
        raise RuntimeError(
            f"Hugging Face revision mismatch: expected={revision} actual={resolved.name}"
        )
    missing = [name for name in required_files if not (resolved / name).is_file()]
    if missing:
        raise RuntimeError(
            f"pinned TTS snapshot {resolved} lacks runtime artifacts: {missing}"
        )
    return resolved


def prefetch_tts(
    root: Path,
    profile_name: str,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> dict[str, dict[str, str]]:
    manifest = load_manifest(root)
    resolved_name, profile = load_profile(root, profile_name)
    environment = service_environments(
        root, manifest, resolved_name, profile, {}
    )["tts"]
    hf_home = Path(environment["HF_HOME"]).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)

    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        snapshot_download = hf_snapshot_download

    tts = manifest["components"]["tts"]
    specs = {
        "backbone": (
            str(tts["backbone_model_id"]),
            str(tts["backbone_revision"]),
            BACKBONE_FILES,
        ),
        "codec": (
            str(tts["codec_model_id"]),
            str(tts["codec_revision"]),
            CODEC_FILES,
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (model_id, revision, required_files) in specs.items():
        downloaded = snapshot_download(
            repo_id=model_id,
            revision=revision,
            repo_type="model",
            cache_dir=str(hf_home / "hub"),
            allow_patterns=list(required_files),
        )
        snapshot = _verify_snapshot(
            Path(downloaded), revision, required_files=required_files
        )
        result[name] = {
            "model_id": model_id,
            "revision": revision,
            "snapshot": str(snapshot),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch and verify pinned VieNeu Hugging Face artifacts"
    )
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    manifest = load_manifest(ROOT)
    profile_name, profile = load_profile(ROOT, args.profile)
    hf_home = service_environments(
        ROOT, manifest, profile_name, profile, {}
    )["tts"]["HF_HOME"]
    artifacts = prefetch_tts(ROOT, profile_name)
    print(f"HF_HOME {hf_home}")
    for name, artifact in artifacts.items():
        print(
            f"TTS {name} PINNED {artifact['model_id']}@{artifact['revision']} "
            f"snapshot={artifact['snapshot']}"
        )


if __name__ == "__main__":
    main()
