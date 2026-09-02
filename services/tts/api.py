from __future__ import annotations

import importlib.metadata
import inspect
import os
import statistics
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from vieneu import Vieneu
except ImportError:
    Vieneu = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "48000"))
CHANNELS = 1
DTYPE = "float32"
VIENEU_MODE = os.getenv("TTS_VIENEU_MODE", "v3turbo").strip() or "v3turbo"
TTS_PROFILE = "realtime"
VIENEU_BACKEND = "onnx"
VIENEU_DEVICE = "cpu"
DEFAULT_GENERATION_MODE = "stream"

# V10 app units are capped at 176 chars by default. Keep the upstream ceiling
# above that so VieNeu does not silently re-split a semantic unit internally.
STREAM_MAX_CHARS = max(176, int(os.getenv("TTS_STREAM_MAX_CHARS", "192")))
SENTENCE_MAX_CHARS = max(64, int(os.getenv("TTS_SENTENCE_MAX_CHARS", "512")))
FORCE_BLOCKING = os.getenv("TTS_FORCE_BLOCKING", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
WARMUP_ENABLED = os.getenv("TTS_WARMUP", "true").strip().lower() not in {
    "0", "false", "no", "off"
}
WARMUP_RUNS = max(1, int(os.getenv("TTS_WARMUP_RUNS", "2")))
WARMUP_VOICE_INDEX = max(0, int(os.getenv("TTS_WARMUP_VOICE_INDEX", "2")))
WARMUP_TEXT = os.getenv(
    "TTS_WARMUP_TEXT",
    "Xin chào, hệ thống đang khởi động giọng nói thời gian thực.",
).strip()
UPSTREAM_COMMIT = os.getenv("VIENEU_UPSTREAM_COMMIT", "unknown")
UPSTREAM_REPOSITORY = os.getenv(
    "VIENEU_UPSTREAM_REPOSITORY",
    "https://github.com/pnnbao97/VieNeu-TTS.git",
)
TTS_BACKBONE_MODEL_ID = os.getenv("TTS_BACKBONE_MODEL_ID")
TTS_BACKBONE_REVISION = os.getenv("TTS_BACKBONE_REVISION")
TTS_CODEC_MODEL_ID = os.getenv("TTS_CODEC_MODEL_ID")
TTS_CODEC_REVISION = os.getenv("TTS_CODEC_REVISION")

model = None
voices: list[Any] = []
load_error: str | None = None
load_time_seconds: float | None = None
streaming_supported = False
resolved_backend: str | None = None
warmup_metrics: dict[str, Any] | None = None
resolved_model_storage_root: str | None = None

# Optional trace framing used by the V10 application client.  HTTP is allowed
# to split or coalesce body writes, so native generator yields cannot otherwise
# be correlated with client reads.  Benchmark/legacy callers keep raw float32.
_TRACE_MAGIC = b"TTS1"
_TRACE_HEADER = struct.Struct("<4sIIIQQ")


def _cached_huggingface_revision(model_id: str | None) -> str | None:
    if not model_id:
        return None
    cache_root = Path(os.getenv("HF_HOME", "")) / "hub"
    ref = cache_root / ("models--" + model_id.replace("/", "--")) / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _voice_identifier(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[1]
    if isinstance(value, dict):
        for key in ("id", "voice_id", "name"):
            if value.get(key):
                return value[key]
    return value


def _supported_kwargs(callable_obj, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return values
    parameters = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def _as_float32_mono(audio: Any) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("VieNeu returned an empty waveform chunk")
    if not np.isfinite(samples).all():
        raise RuntimeError("VieNeu returned NaN/Inf audio")
    return samples


def _resolve_generation_mode(requested: str) -> str:
    mode = (requested or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"auto", "stream", "sentence"}:
        raise ValueError(f"unsupported generation_mode={mode!r}")
    if mode == "auto":
        mode = "stream"
    if mode == "stream" and FORCE_BLOCKING:
        return "sentence"
    return mode


def _model_init_kwargs() -> dict[str, Any]:
    # Match VieNeu's published realtime path exactly: Vieneu(backend="onnx").
    # Do not pass device="auto" on CUDA hosts.
    return {"mode": VIENEU_MODE, "backend": "onnx"}


def iter_model_audio_traced(
    text: str,
    *,
    selected_voice: Any,
    pre_chunked: bool,
    generation_mode: str = "stream",
) -> Iterator[tuple[np.ndarray, int]]:
    if model is None:
        raise RuntimeError("TTS model is not ready")

    mode = _resolve_generation_mode(generation_mode)
    if mode == "sentence":
        infer_fn = model.infer
        kwargs = _supported_kwargs(
            infer_fn,
            {"text": text, "voice": selected_voice, "max_chars": SENTENCE_MAX_CHARS},
        )
        samples = infer_fn(**kwargs)
        model_yield_ns = time.time_ns()
        yield _as_float32_mono(samples), model_yield_ns
        return

    stream_fn = getattr(model, "infer_stream", None)
    if not callable(stream_fn) or FORCE_BLOCKING:
        raise RuntimeError("V10 realtime requires VieNeu infer_stream()")

    kwargs = _supported_kwargs(
        stream_fn,
        {"text": text, "voice": selected_voice, "max_chars": STREAM_MAX_CHARS},
    )
    yielded = False
    for audio_chunk in stream_fn(**kwargs):
        model_yield_ns = time.time_ns()
        if audio_chunk is None:
            continue
        samples = np.asarray(audio_chunk)
        if samples.size:
            yielded = True
            yield _as_float32_mono(samples), model_yield_ns
    if not yielded:
        raise RuntimeError("VieNeu infer_stream() produced no audio")


def iter_model_audio(
    text: str,
    *,
    selected_voice: Any,
    pre_chunked: bool,
    generation_mode: str = "stream",
) -> Iterator[np.ndarray]:
    """Compatibility iterator returning only PCM samples."""
    for samples, _model_yield_ns in iter_model_audio_traced(
        text,
        selected_voice=selected_voice,
        pre_chunked=pre_chunked,
        generation_mode=generation_mode,
    ):
        yield samples


def _measure_stream(text: str, selected_voice: Any) -> dict[str, Any]:
    started = time.perf_counter()
    first_at: float | None = None
    last_at: float | None = None
    gaps: list[float] = []
    sample_count = 0
    chunk_count = 0

    for chunk in iter_model_audio(
        text,
        selected_voice=selected_voice,
        pre_chunked=True,
        generation_mode="stream",
    ):
        now = time.perf_counter()
        if first_at is None:
            first_at = now
        if last_at is not None:
            gaps.append((now - last_at) * 1000.0)
        last_at = now
        sample_count += int(chunk.size)
        chunk_count += 1

    finished = time.perf_counter()
    audio_ms = sample_count / float(SAMPLE_RATE) * 1000.0
    total_ms = (finished - started) * 1000.0
    return {
        "ttfb_ms": (first_at - started) * 1000.0 if first_at is not None else None,
        "total_ms": total_ms,
        "audio_duration_ms": audio_ms,
        "rtf": total_ms / audio_ms if audio_ms > 0 else None,
        "chunk_count": chunk_count,
        "max_inter_chunk_gap_ms": max(gaps) if gaps else None,
        "p95_inter_chunk_gap_ms": (
            statistics.quantiles(gaps, n=20, method="inclusive")[18]
            if len(gaps) > 1
            else (gaps[0] if gaps else None)
        ),
    }


def _run_warmup() -> dict[str, Any]:
    if not voices:
        raise RuntimeError("No preset voices available for warmup")
    voice_index = min(WARMUP_VOICE_INDEX, len(voices) - 1)
    selected_voice = model.get_preset_voice(_voice_identifier(voices[voice_index]))

    # V10 primes both interactive-short and steady-state unit shapes. V9 only
    # warmed one short sentence, while the first real 90-char unit still paid a
    # large cold/shape cost on Colab.
    texts = [
        WARMUP_TEXT,
        os.getenv(
            "TTS_WARMUP_MEDIUM_TEXT",
            "Trợ lý đang chuẩn bị một đoạn phát âm dài vừa phải để ổn định bộ nhớ và đường suy luận ONNX trước lượt hội thoại đầu tiên.",
        ).strip(),
    ]
    runs: list[dict[str, Any]] = []
    for text in texts:
        for _ in range(WARMUP_RUNS):
            row = _measure_stream(text, selected_voice)
            row["chars"] = len(text)
            runs.append(row)

    rtfs = [float(x["rtf"]) for x in runs if x.get("rtf") is not None]
    ttfbs = [float(x["ttfb_ms"]) for x in runs if x.get("ttfb_ms") is not None]
    gaps = [float(x["max_inter_chunk_gap_ms"]) for x in runs if x.get("max_inter_chunk_gap_ms") is not None]
    median_rtf = statistics.median(rtfs) if rtfs else None
    median_ttfb = statistics.median(ttfbs) if ttfbs else None
    max_gap = max(gaps) if gaps else None

    if median_rtf is None or median_ttfb is None:
        performance = "unknown"
    elif median_rtf < 1.0 and median_ttfb < 1200 and (max_gap is None or max_gap < 750):
        performance = "realtime-headroom"
    else:
        performance = "degraded"
    return {
        "enabled": True,
        "voice_index": voice_index,
        "profiles": ["short", "medium"],
        "runs": runs,
        "median_rtf": median_rtf,
        "median_ttfb_ms": median_ttfb,
        "max_inter_chunk_gap_ms": max_gap,
        "performance": performance,
    }

def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, voices, load_error, load_time_seconds
    global streaming_supported, resolved_backend, warmup_metrics
    global resolved_model_storage_root

    if Vieneu is None:
        load_error = "Missing dependency 'vieneu'"
        yield
        return

    started = time.perf_counter()
    try:
        model_path = os.getenv("MODEL_PATH", "./models/vieneu-tts-v3-turbo")
        os.environ.setdefault("HF_HOME", model_path)
        resolved_model_storage_root = str(
            Path(os.environ["HF_HOME"]).expanduser().resolve(strict=False)
        )

        model = Vieneu(**_model_init_kwargs())
        voices = list(model.list_preset_voices())
        load_time_seconds = time.perf_counter() - started
        streaming_supported = callable(getattr(model, "infer_stream", None)) and not FORCE_BLOCKING
        resolved_backend = str(getattr(model, "backend", VIENEU_BACKEND)).lower()

        if "onnx" not in resolved_backend:
            raise RuntimeError(
                "V10 realtime requires VieNeu ONNX backend; "
                f"resolved backend was {resolved_backend!r}"
            )
        if not streaming_supported:
            raise RuntimeError("V10 realtime requires VieNeu infer_stream()")

        warmup_metrics = {"enabled": False}
        if WARMUP_ENABLED:
            warmup_metrics = _run_warmup()

        load_error = None
        print(
            f"[Boot] VieNeu V10 ready in {load_time_seconds:.2f}s; "
            f"backend={resolved_backend}; native_stream={streaming_supported}; "
            f"upstream_commit={UPSTREAM_COMMIT}; warmup={warmup_metrics.get('performance')}"
        )
    except Exception as exc:
        model = None
        voices = []
        streaming_supported = False
        resolved_backend = None
        warmup_metrics = None
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"[Boot] VieNeu V10 load failed: {load_error}")

    yield


app = FastAPI(title="Eureka V10 TTS API", version="0.10.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    voice_index: int = Field(default=2, ge=0)
    pre_chunked: bool = True
    generation_mode: Literal["auto", "stream", "sentence"] = "stream"
    trace_framed: bool = False
    trace_id: str | None = Field(default=None, max_length=128)
    unit_index: int | None = Field(default=None, ge=1)


@app.get("/health/live")
def health_live():
    return {"status": "alive", "profile": TTS_PROFILE, "pipeline": "v10"}


@app.get("/health/ready")
def health_ready():
    if model is None or not voices:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "profile": TTS_PROFILE, "error": load_error or "model not loaded"},
        )
    package_path = None
    try:
        package_path = str(Path(inspect.getfile(Vieneu)).resolve())
    except Exception:
        pass
    return {
        "status": "ready",
        "pipeline": "v10",
        "profile": TTS_PROFILE,
        "voice_count": len(voices),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "dtype": DTYPE,
        "model_load_time_seconds": round(load_time_seconds, 3) if load_time_seconds is not None else None,
        "streaming_supported": streaming_supported,
        "streaming_mode": "frame-pcm" if streaming_supported else "unavailable",
        "backend": resolved_backend,
        "device_policy": "cpu/onnx-pinned",
        "runtime_profile": os.getenv("RUNTIME_PROFILE"),
        "model_manifest_id": os.getenv("MODEL_MANIFEST_ID"),
        "model_storage_root": resolved_model_storage_root,
        "model_artifacts": {
            "backbone": {
                "model_id": TTS_BACKBONE_MODEL_ID,
                "requested_revision": TTS_BACKBONE_REVISION,
                "resolved_revision": _cached_huggingface_revision(TTS_BACKBONE_MODEL_ID),
            },
            "codec": {
                "model_id": TTS_CODEC_MODEL_ID,
                "requested_revision": TTS_CODEC_REVISION,
                "resolved_revision": _cached_huggingface_revision(TTS_CODEC_MODEL_ID),
            },
        },
        "stream_max_chars": STREAM_MAX_CHARS,
        "default_generation_mode": DEFAULT_GENERATION_MODE,
        "recommended_integration_mode": "decoupled-speech-units-websocket-audioworklet-v10",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "vieneu_version": _package_version("vieneu"),
            "onnxruntime_version": _package_version("onnxruntime"),
            "onnxruntime_providers": (ort.get_available_providers() if ort is not None else None),
            "cpu_count": os.cpu_count(),
            "process_id": os.getpid(),
            "python_thread_count": threading.active_count(),
            "thread_environment": {
                key: os.getenv(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OMP_WAIT_POLICY",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "ORT_NUM_THREADS",
                )
            },
            "package_path": package_path,
            "environment_policy": "uv-sync-frozen",
        },
        "warmup": warmup_metrics,
    }


@app.post("/api/tts")
def generate_tts(request: TTSRequest):
    if model is None or not voices:
        raise HTTPException(status_code=503, detail=load_error or "TTS model is not ready")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    if request.voice_index >= len(voices):
        raise HTTPException(status_code=422, detail=f"voice_index must be in [0, {len(voices) - 1}]")

    try:
        selected_voice = model.get_preset_voice(_voice_identifier(voices[request.voice_index]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not resolve preset voice: {exc}") from exc

    generation_mode = _resolve_generation_mode(request.generation_mode)

    def audio_stream_generator() -> Iterator[bytes]:
        sequence = 0
        for samples, model_yield_ns in iter_model_audio_traced(
            text,
            selected_voice=selected_voice,
            pre_chunked=bool(request.pre_chunked),
            generation_mode=generation_mode,
        ):
            raw = samples.astype("<f4", copy=False).tobytes()
            if request.trace_framed:
                sequence += 1
                service_yield_ns = time.time_ns()
                yield _TRACE_HEADER.pack(
                    _TRACE_MAGIC,
                    sequence,
                    int(samples.size),
                    len(raw),
                    model_yield_ns,
                    service_yield_ns,
                ) + raw
            else:
                yield raw

    return StreamingResponse(
        audio_stream_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": str(CHANNELS),
            "X-Audio-Dtype": DTYPE,
            "X-Audio-Streaming": "frame-pcm" if generation_mode == "stream" else "sentence-buffered",
            "X-TTS-Generation-Mode": generation_mode,
            "X-TTS-Backend": resolved_backend or "unknown",
            "X-TTS-Profile": TTS_PROFILE,
            "X-VieNeu-Upstream-Commit": UPSTREAM_COMMIT,
            "X-TTS-Trace-Protocol": "tts1-f32" if request.trace_framed else "none",
            "X-TTS-Trace-Id": request.trace_id or "",
            "X-TTS-Unit-Index": str(request.unit_index or ""),
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
