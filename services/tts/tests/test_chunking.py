import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import api
from api import iter_model_audio, iter_model_audio_traced


def test_iter_model_audio_auto_uses_native_stream(monkeypatch):
    class FakeModel:
        def infer_stream(self, **kwargs):
            yield np.ones(120, dtype=np.float32) * 0.1
            yield np.ones(80, dtype=np.float32) * 0.2

        def infer(self, **kwargs):
            raise AssertionError("blocking infer should not be used")

    monkeypatch.setattr(api, "model", FakeModel())
    monkeypatch.setattr(api, "FORCE_BLOCKING", False)
    chunks = list(
        iter_model_audio(
            "Xin chào",
            selected_voice={"voice": "x"},
            pre_chunked=True,
            generation_mode="auto",
        )
    )
    assert [chunk.size for chunk in chunks] == [120, 80]


def test_traced_iterator_marks_each_native_model_yield(monkeypatch):
    class FakeModel:
        def infer_stream(self, **kwargs):
            yield np.ones(12, dtype=np.float32)
            yield np.ones(8, dtype=np.float32)

    monkeypatch.setattr(api, "model", FakeModel())
    monkeypatch.setattr(api, "FORCE_BLOCKING", False)
    rows = list(
        iter_model_audio_traced(
            "Xin chào",
            selected_voice={"voice": "x"},
            pre_chunked=True,
            generation_mode="stream",
        )
    )
    assert [samples.size for samples, _ in rows] == [12, 8]
    assert all(isinstance(yield_ns, int) and yield_ns > 0 for _, yield_ns in rows)


def test_sentence_mode_remains_explicit_compatibility_path(monkeypatch):
    calls = []

    class FakeModel:
        def infer(self, **kwargs):
            calls.append(kwargs)
            return np.ones(256, dtype=np.float32) * 0.1

    monkeypatch.setattr(api, "model", FakeModel())
    text = "Đây là một câu hoàn chỉnh."
    chunks = list(
        iter_model_audio(
            text,
            selected_voice={"voice": "x"},
            pre_chunked=True,
            generation_mode="sentence",
        )
    )
    assert len(chunks) == 1
    assert calls[0]["text"] == text
    assert calls[0]["max_chars"] == api.SENTENCE_MAX_CHARS


def test_v10_realtime_rejects_missing_native_stream(monkeypatch):
    class FakeModel:
        def infer(self, **kwargs):
            return np.ones(64, dtype=np.float32)

    monkeypatch.setattr(api, "model", FakeModel())
    monkeypatch.setattr(api, "FORCE_BLOCKING", False)
    with pytest.raises(RuntimeError, match="requires VieNeu infer_stream"):
        list(
            iter_model_audio(
                "Xin chào",
                selected_voice={"voice": "x"},
                pre_chunked=True,
                generation_mode="stream",
            )
        )


def test_v10_runtime_is_onnx_streaming_profile(monkeypatch):
    assert api.TTS_PROFILE == "realtime"
    assert api.VIENEU_BACKEND == "onnx"
    assert api.VIENEU_DEVICE == "cpu"
    assert api.DEFAULT_GENERATION_MODE == "stream"
    assert 176 <= api.STREAM_MAX_CHARS <= 224
    monkeypatch.setattr(api, "TTS_BACKBONE_MODEL_ID", None)
    monkeypatch.setattr(api, "TTS_BACKBONE_REVISION", None)
    assert api._model_init_kwargs() == {"mode": api.VIENEU_MODE, "backend": "onnx"}


def test_model_init_uses_exact_pinned_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "TTS_BACKBONE_MODEL_ID", "org/model")
    monkeypatch.setattr(api, "TTS_BACKBONE_REVISION", "abc123")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / "abc123"
    (snapshot / "onnx_int8").mkdir(parents=True)
    assert api._model_init_kwargs()["backbone_repo"] == str(snapshot.resolve())
    assert api._model_init_kwargs()["onnx_dir"] == str((snapshot / "onnx_int8").resolve())


def test_missing_pinned_snapshot_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "TTS_BACKBONE_MODEL_ID", "org/model")
    monkeypatch.setattr(api, "TTS_BACKBONE_REVISION", "missing")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="Pinned TTS snapshot is missing"):
        api._model_init_kwargs()


def test_codec_hub_download_is_forced_to_manifest_revision(monkeypatch, tmp_path):
    calls = []
    codec_revision = "codec-commit"

    def fake_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        path = tmp_path / "snapshots" / kwargs["revision"] / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    fake_hub = SimpleNamespace(hf_hub_download=fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(api, "TTS_BACKBONE_MODEL_ID", None)
    monkeypatch.setattr(api, "TTS_BACKBONE_REVISION", None)
    monkeypatch.setattr(api, "TTS_CODEC_MODEL_ID", "org/codec")
    monkeypatch.setattr(api, "TTS_CODEC_REVISION", codec_revision)
    monkeypatch.setattr(api, "enforced_download_revisions", {})

    with api._pinned_huggingface_downloads():
        downloaded = fake_hub.hf_hub_download("org/codec", "codec.onnx")

    assert Path(downloaded).parts[-2] == codec_revision
    assert calls[0][2]["revision"] == codec_revision
    assert api.enforced_download_revisions == {"org/codec": codec_revision}
    assert fake_hub.hf_hub_download is fake_download


def test_warmup_measurement_reports_rtf_and_gaps(monkeypatch):
    class FakeModel:
        def infer_stream(self, **kwargs):
            yield np.ones(480, dtype=np.float32) * 0.1
            yield np.ones(480, dtype=np.float32) * 0.1

    monkeypatch.setattr(api, "model", FakeModel())
    monkeypatch.setattr(api, "SAMPLE_RATE", 48000)
    result = api._measure_stream("Xin chào", {"voice": "x"})
    assert result["chunk_count"] == 2
    assert result["audio_duration_ms"] == pytest.approx(20.0)
    assert result["rtf"] is not None
    assert result["ttfb_ms"] is not None



def test_v10_warmup_profiles_short_and_medium(monkeypatch):
    class FakeModel:
        def get_preset_voice(self, x): return x
        def infer_stream(self, **kwargs):
            yield np.ones(480, dtype=np.float32) * 0.1
    monkeypatch.setattr(api, "model", FakeModel())
    monkeypatch.setattr(api, "voices", [("x", "x")])
    monkeypatch.setattr(api, "WARMUP_RUNS", 1)
    result = api._run_warmup()
    assert result["profiles"] == ["short", "medium"]
    assert len(result["runs"]) == 2
