import io
import os
import time
import asyncio
from typing import Any

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from runtime import normalize_model_inputs, safe_max_new_tokens

try:
    from silero_vad import load_silero_vad, VADIterator
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency 'silero-vad'. Install it with: pip install silero-vad"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = os.getenv("MODEL_ID", "vinai/PhoWhisper-base")
MODEL_REVISION = os.getenv("MODEL_REVISION", "").strip() or None
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "").strip() or None
MODEL_LOCAL_FILES_ONLY = os.getenv("MODEL_LOCAL_FILES_ONLY", "false").strip().casefold() in {
    "1", "true", "yes", "on"
}

REQUESTED_DEVICE = os.getenv("MODEL_DEVICE", "auto").strip().casefold() or "auto"
if REQUESTED_DEVICE not in {"auto", "cpu", "cuda"}:
    raise RuntimeError("MODEL_DEVICE must be one of: auto, cpu, cuda")
if REQUESTED_DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("MODEL_DEVICE=cuda was requested but CUDA is unavailable")
DEVICE = (
    "cuda" if REQUESTED_DEVICE == "cuda"
    else "cpu" if REQUESTED_DEVICE == "cpu"
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

REQUESTED_DTYPE = os.getenv("MODEL_DTYPE", "auto").strip().casefold() or "auto"
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
if REQUESTED_DTYPE == "auto":
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
elif REQUESTED_DTYPE in _DTYPES:
    DTYPE = _DTYPES[REQUESTED_DTYPE]
else:
    raise RuntimeError("MODEL_DTYPE must be one of: auto, float32, float16, bfloat16")
if DEVICE == "cpu" and DTYPE == torch.float16:
    raise RuntimeError("MODEL_DTYPE=float16 is not supported by the certified STT CPU profile")

TARGET_SR = 16000

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"
}

# Silero VAD settings
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "500"))
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "80"))

# Safety limit for a single utterance
MAX_UTTERANCE_SECONDS = float(
    os.getenv("MAX_UTTERANCE_SECONDS", "30")
)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PhoWhisper-base Vietnamese STT API",
    description=(
        "Vietnamese Speech-to-Text API using vinai/PhoWhisper-base "
        "with Silero VAD utterance-level WebSocket streaming."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

print("=" * 60)
print("Loading PhoWhisper-base")
print("Model :", MODEL_ID)
print("Device:", DEVICE)
print("Dtype :", DTYPE)

load_start = time.perf_counter()

model_load_kwargs = {
    "revision": MODEL_REVISION,
    "cache_dir": MODEL_CACHE_DIR,
    "local_files_only": MODEL_LOCAL_FILES_ONLY,
}
model_load_kwargs = {
    key: value for key, value in model_load_kwargs.items() if value is not None
}

processor = AutoProcessor.from_pretrained(MODEL_ID, **model_load_kwargs)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    **model_load_kwargs,
).to(DEVICE)

model.eval()

load_time = time.perf_counter() - load_start

print(f"PhoWhisper loaded in {load_time:.2f}s")
print("=" * 60)

print("Loading Silero VAD")
vad_load_start = time.perf_counter()

# Keep VAD on CPU so it does not compete with PhoWhisper for GPU memory.
vad_model = load_silero_vad(onnx=False)

vad_load_time = time.perf_counter() - vad_load_start

print(f"Silero VAD loaded in {vad_load_time:.2f}s")
print("=" * 60)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(data: bytes):
    """Decode uploaded audio and resample it to 16 kHz mono."""
    try:
        audio, sr = librosa.load(
            io.BytesIO(data),
            sr=TARGET_SR,
            mono=True,
        )
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc

    if len(audio) == 0:
        raise ValueError("Audio contains no samples.")

    return np.asarray(audio, dtype=np.float32), TARGET_SR


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """
    Convert raw PCM16 little-endian mono audio to float32.

    WebSocket client format:
        PCM signed 16-bit / mono / 16000 Hz / little-endian
    """
    if not data:
        return np.empty(0, dtype=np.float32)

    if len(data) % 2 != 0:
        raise ValueError("PCM16 audio chunk has an odd number of bytes.")

    pcm = np.frombuffer(data, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def transcribe_waveform(audio: np.ndarray, sample_rate: int):
    """Run PhoWhisper on one complete utterance."""
    if audio is None or len(audio) == 0:
        raise ValueError("Audio contains no samples.")

    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / sample_rate

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    inputs = normalize_model_inputs(inputs, model)
    max_new_tokens = safe_max_new_tokens(model, MAX_NEW_TOKENS, reserve_tokens=8)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            language="vi",
            task="transcribe",
            max_new_tokens=max_new_tokens,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start

    text = processor.batch_decode(
        ids,
        skip_special_tokens=True,
    )[0].strip()

    rtf = inference_time / duration if duration > 0 else None

    return text, inference_time, duration, rtf


# ---------------------------------------------------------------------------
# VAD helpers
# ---------------------------------------------------------------------------

def create_vad_iterator() -> VADIterator:
    """Create an independent VAD state for one WebSocket connection."""
    return VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=TARGET_SR,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )


def reset_vad(vad_iterator: VADIterator) -> None:
    vad_iterator.reset_states()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Vietnamese Speech-to-Text API",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": DEVICE,
        "status": "running",
        "streaming": True,
        "vad": "Silero VAD",
        "streaming_mode": "utterance-level",
        "docs": "/docs",
    }


@app.get("/health/live")
def health_live() -> dict[str, Any]:
    return {"status": "alive"}


@app.get("/health/ready")
@app.get("/health")
def health() -> dict[str, Any]:
    resolved_revision = (
        getattr(getattr(model, "config", None), "_commit_hash", None)
        or MODEL_REVISION
    )
    return {
        "status": "ok",
        "model": MODEL_ID,
        "revision": resolved_revision,
        "requested_revision": MODEL_REVISION,
        "device": DEVICE,
        "requested_device": REQUESTED_DEVICE,
        "dtype": str(DTYPE).removeprefix("torch."),
        "requested_dtype": REQUESTED_DTYPE,
        "local_files_only": MODEL_LOCAL_FILES_ONLY,
        "cache_dir": MODEL_CACHE_DIR or os.getenv("HF_HOME"),
        "runtime_profile": os.getenv("RUNTIME_PROFILE"),
        "model_manifest_id": os.getenv("MODEL_MANIFEST_ID"),
        "cuda_available": torch.cuda.is_available(),
        "model_load_time_seconds": round(load_time, 3),
        "vad_load_time_seconds": round(vad_load_time, 3),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Existing file-based endpoint kept for normal testing."""
    filename = file.filename or "audio"
    extension = os.path.splitext(filename)[1].lower()

    if extension and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported format '{extension}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        audio, sr = load_audio(data)
        text, inference_time, duration, rtf = transcribe_waveform(
            audio, sr
        )

        return {
            "success": True,
            "model": MODEL_ID,
            "text": text,
            "audio": {
                "filename": filename,
                "sample_rate": sr,
                "duration_seconds": round(duration, 3),
            },
            "inference": {
                "time_seconds": round(inference_time, 3),
                "rtf": round(rtf, 5) if rtf is not None else None,
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# WebSocket utterance-level streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    Client sends raw PCM16 little-endian, mono, 16000 Hz chunks.

    Server events:
        ready
        started
        speech_start
        speech_end
        transcript
        error
        session_end

    Commands:
        start
        stop
        ping
    """
    await websocket.accept()

    vad_iterator = create_vad_iterator()

    utterance_chunks: list[np.ndarray] = []
    utterance_samples = 0
    in_speech = False

    transcription_lock = asyncio.Lock()

    async def send_event(payload: dict[str, Any]) -> None:
        await websocket.send_json(payload)

    async def transcribe_current_utterance(reason: str) -> None:
        nonlocal utterance_chunks
        nonlocal utterance_samples
        nonlocal in_speech

        if not utterance_chunks or utterance_samples <= 0:
            utterance_chunks = []
            utterance_samples = 0
            in_speech = False
            return

        audio = np.concatenate(utterance_chunks).astype(
            np.float32,
            copy=False,
        )

        utterance_chunks = []
        utterance_samples = 0
        in_speech = False

        duration = len(audio) / TARGET_SR

        await send_event(
            {
                "event": "speech_end",
                "reason": reason,
                "duration_seconds": round(duration, 3),
            }
        )

        async with transcription_lock:
            try:
                text, inference_time, duration, rtf = await asyncio.to_thread(
                    transcribe_waveform,
                    audio,
                    TARGET_SR,
                )

                await send_event(
                    {
                        "event": "transcript",
                        "success": True,
                        "model": MODEL_ID,
                        "text": text,
                        "audio": {
                            "sample_rate": TARGET_SR,
                            "duration_seconds": round(duration, 3),
                        },
                        "inference": {
                            "time_seconds": round(inference_time, 3),
                            "rtf": round(rtf, 5)
                            if rtf is not None
                            else None,
                        },
                    }
                )

            except Exception as exc:
                print(f"[WS ERROR] Transcription failed: {exc}")

                await send_event(
                    {
                        "event": "transcript",
                        "success": False,
                        "text": "",
                        "error": str(exc),
                    }
                )

        reset_vad(vad_iterator)

    try:
        await send_event(
            {
                "event": "ready",
                "model": MODEL_ID,
                "sample_rate": TARGET_SR,
                "audio_format": "pcm_s16le",
                "channels": 1,
                "vad": "silero-vad",
                "streaming_mode": "utterance-level",
            }
        )

        while True:
            message = await websocket.receive()

            # ---------------------------------------------------------------
            # Text/control messages
            # ---------------------------------------------------------------
            if message.get("text") is not None:
                command = message["text"].strip().lower()

                if command == "start":
                    utterance_chunks = []
                    utterance_samples = 0
                    in_speech = False
                    reset_vad(vad_iterator)

                    await send_event({"event": "started"})
                    continue

                if command == "stop":
                    await transcribe_current_utterance(
                        reason="client_stop"
                    )

                    await send_event({"event": "session_end"})
                    break

                if command == "ping":
                    await send_event({"event": "pong"})
                    continue

                await send_event(
                    {
                        "event": "error",
                        "error": (
                            "Unknown command. "
                            "Supported commands: start, stop, ping."
                        ),
                    }
                )
                continue

            # ---------------------------------------------------------------
            # Binary audio
            # ---------------------------------------------------------------
            data = message.get("bytes")

            if data is None or not data:
                continue

            try:
                chunk = pcm16_bytes_to_float32(data)
            except ValueError as exc:
                await send_event(
                    {
                        "event": "error",
                        "error": str(exc),
                    }
                )
                continue

            if len(chunk) == 0:
                continue

            chunk_tensor = torch.from_numpy(chunk)
            vad_event = vad_iterator(chunk_tensor)

            if vad_event and "start" in vad_event:
                in_speech = True

                await send_event(
                    {
                        "event": "speech_start",
                        "timestamp_seconds": round(
                            vad_event["start"] / TARGET_SR,
                            3,
                        ),
                    }
                )

            if in_speech:
                utterance_chunks.append(chunk)
                utterance_samples += len(chunk)

                if (
                    utterance_samples / TARGET_SR
                    >= MAX_UTTERANCE_SECONDS
                ):
                    await transcribe_current_utterance(
                        reason="max_utterance_duration"
                    )
                    continue

            if vad_event and "end" in vad_event:
                await transcribe_current_utterance(
                    reason="vad_silence"
                )

    except WebSocketDisconnect:
        if utterance_chunks:
            try:
                await transcribe_current_utterance(
                    reason="client_disconnect"
                )
            except Exception as exc:
                print(f"[WS DISCONNECT ERROR] {exc}")

    except Exception as exc:
        print(f"[WS ERROR] {exc}")

        try:
            await websocket.send_json(
                {
                    "event": "error",
                    "error": str(exc),
                }
            )
        except Exception:
            pass

    finally:
        try:
            reset_vad(vad_iterator)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Local development entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
