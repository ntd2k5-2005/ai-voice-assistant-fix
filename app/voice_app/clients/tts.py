from __future__ import annotations

import time
import uuid
import struct
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
from scipy.io import wavfile

from ..contracts import ServiceStatus, TTSResult
from ..errors import ContractError, ServiceError
from ..streaming import pcm_to_wav_bytes
from .base import clean_base_url, expect_json_object, response_error


_DTYPE_MAP = {
    "float32": np.dtype("<f4"),
    "int16": np.dtype("<i2"),
}
_TRACE_MAGIC = b"TTS1"
_TRACE_HEADER = struct.Struct("<4sIIIQQ")


def decode_raw_pcm(
    raw: bytes,
    *,
    dtype: str,
    channels: int,
) -> np.ndarray:
    if dtype not in _DTYPE_MAP:
        raise ContractError(
            f"TTS: unsupported configured dtype {dtype!r}"
        )
    if channels < 1:
        raise ContractError("TTS: channels must be >= 1")

    np_dtype = _DTYPE_MAP[dtype]
    frame_bytes = np_dtype.itemsize * channels
    if not raw:
        raise ContractError(
            "TTS: upstream returned an empty audio stream"
        )
    if len(raw) % frame_bytes != 0:
        raise ContractError(
            "TTS: raw byte length is not aligned to configured "
            f"dtype/channels (bytes={len(raw)}, "
            f"frame_bytes={frame_bytes})"
        )

    samples = np.frombuffer(raw, dtype=np_dtype)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    if dtype == "float32" and not np.isfinite(samples).all():
        raise ContractError("TTS: audio contains NaN/Inf values")
    return samples


def write_wav(
    path: Path,
    sample_rate: int,
    samples: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, samples)


class TTSClient:
    service_name = "TTS"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 180.0,
        sample_rate: int = 48000,
        channels: int = 1,
        dtype: str = "float32",
        stream_chunk_ms: int = 250,
        generation_mode: str = "sentence",
        reject_all_silence: bool = True,
        output_dir: str | Path = "/tmp/voice_app_audio",
        keep_last_files: int = 50,
    ) -> None:
        self.base_url = clean_base_url(base_url)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = dtype
        self.stream_chunk_ms = max(50, int(stream_chunk_ms))
        self.generation_mode = generation_mode.strip().lower()
        if self.generation_mode not in {"sentence", "stream", "auto"}:
            raise ValueError("generation_mode must be sentence, stream, or auto")
        self.reject_all_silence = reject_all_silence
        self.output_dir = Path(output_dir)
        self.keep_last_files = max(1, int(keep_last_files))
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_s)
        )

    def close(self) -> None:
        self.client.close()

    def health(self) -> ServiceStatus:
        started = time.perf_counter()
        try:
            r = self.client.get(
                f"{self.base_url}/health/ready"
            )
            latency = (time.perf_counter() - started) * 1000

            if r.status_code == 200:
                data = expect_json_object(self.service_name, r)
                return ServiceStatus(
                    service=self.service_name,
                    reachable=True,
                    ready=data.get("status") == "ready",
                    latency_ms=latency,
                    message="VieNeu TTS service ready",
                    details=data,
                )

            if r.status_code == 503:
                return ServiceStatus(
                    service=self.service_name,
                    reachable=True,
                    ready=False,
                    latency_ms=latency,
                    message=(
                        "TTS API reachable but model is not ready"
                    ),
                    details={"body": r.text[:1000]},
                )

            return ServiceStatus(
                service=self.service_name,
                reachable=True,
                ready=False,
                latency_ms=latency,
                message=(
                    "TTS readiness returned "
                    f"HTTP {r.status_code}"
                ),
            )
        except Exception as exc:
            return ServiceStatus(
                service=self.service_name,
                reachable=False,
                ready=False,
                latency_ms=(
                    time.perf_counter() - started
                ) * 1000,
                message=str(exc),
            )

    def _prune_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            self.output_dir.glob("tts-*.wav"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old in files[self.keep_last_files :]:
            try:
                old.unlink()
            except OSError:
                pass

    def _raw_stream(
        self,
        text: str,
        *,
        voice_index: int,
        chunk_ms: int | None = None,
        include_wav: bool = True,
    ) -> Iterator[dict]:
        """Yield browser-sized PCM chunks as soon as upstream frames arrive.

        The previous adapter asked httpx for 192000-byte chunks (~1 second at
        48 kHz float32 mono). That could hide native VieNeu frame streaming by
        buffering several upstream frames before the app saw any audio. We now
        consume raw HTTP chunks immediately and only coalesce them into a small,
        configurable playback quantum (default 250 ms).
        """
        text = text.strip()
        if not text:
            raise ServiceError(self.service_name, "Text is empty")

        np_dtype = _DTYPE_MAP.get(self.dtype)
        if np_dtype is None:
            raise ContractError(
                f"TTS: unsupported configured dtype {self.dtype!r}"
            )
        frame_bytes = np_dtype.itemsize * self.channels
        target_ms = self.stream_chunk_ms if chunk_ms is None else max(50, int(chunk_ms))
        target_bytes = int(
            self.sample_rate
            * self.channels
            * np_dtype.itemsize
            * (target_ms / 1000.0)
        )
        target_bytes = max(
            frame_bytes,
            (target_bytes // frame_bytes) * frame_bytes,
        )

        started = time.perf_counter()
        first_byte_at: float | None = None
        byte_count = 0
        peak = 0.0
        pending = bytearray()
        network_chunk_count = 0
        emitted_chunk_count = 0
        streaming_mode = "unknown"
        generation_mode = "stream"
        upstream_backend = "unknown"

        def make_audio_event(raw: bytes) -> dict:
            nonlocal byte_count, peak, emitted_chunk_count
            samples = decode_raw_pcm(
                raw,
                dtype=self.dtype,
                channels=self.channels,
            )
            if samples.size:
                peak = max(
                    peak,
                    float(np.max(np.abs(samples))),
                )
            byte_count += len(raw)
            emitted_chunk_count += 1
            frames = samples.shape[0] if samples.ndim > 1 else samples.size
            duration_ms = (frames / float(self.sample_rate)) * 1000.0
            event = {
                "type": "audio",
                "raw": raw,
                "duration_ms": duration_ms,
                "chunk_index": emitted_chunk_count,
                "ttfb_ms": (
                    (first_byte_at - started) * 1000
                    if first_byte_at is not None
                    else None
                ),
                "streaming_mode": streaming_mode,
                "upstream_backend": upstream_backend,
            }
            if include_wav:
                event["wav"] = pcm_to_wav_bytes(
                    raw,
                    dtype=self.dtype,
                    channels=self.channels,
                    sample_rate=self.sample_rate,
                )
            return event

        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/api/tts",
                json={
                    "text": text,
                    "voice_index": int(voice_index),
                    "pre_chunked": True,
                    "generation_mode": "stream",
                },
            ) as r:
                if r.status_code != 200:
                    r.read()
                    raise response_error(self.service_name, r)

                content_type = (
                    r.headers.get("content-type") or ""
                ).lower()
                if "application/octet-stream" not in content_type:
                    raise ContractError(
                        "TTS: expected application/octet-stream, got "
                        f"{content_type or '<missing>'}"
                    )

                streaming_mode = r.headers.get("X-Audio-Streaming") or "unknown"
                generation_mode = r.headers.get("X-TTS-Generation-Mode") or "stream"
                upstream_backend = r.headers.get("X-TTS-Backend") or "unknown"

                # iter_raw() preserves low-latency HTTP body delivery. Using
                # iter_bytes(chunk_size=...) would buffer native frame chunks.
                # MockTransport may hand us an already-materialized response;
                # support that case without changing the real network path.
                if r.is_stream_consumed:
                    network_iter = iter((r.content,))
                else:
                    network_iter = r.iter_raw()

                for network_chunk in network_iter:
                    if not network_chunk:
                        continue
                    network_chunk_count += 1
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()

                    pending.extend(network_chunk)
                    while len(pending) >= target_bytes:
                        raw = bytes(pending[:target_bytes])
                        del pending[:target_bytes]
                        yield make_audio_event(raw)

                if pending:
                    aligned = len(pending) - (len(pending) % frame_bytes)
                    if aligned:
                        raw = bytes(pending[:aligned])
                        del pending[:aligned]
                        yield make_audio_event(raw)
                    if pending:
                        raise ContractError(
                            "TTS: trailing non-frame-aligned bytes: "
                            f"{len(pending)}"
                        )

        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            raise ServiceError(
                self.service_name,
                "Synthesis timed out",
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(
                self.service_name,
                f"HTTP error: {exc}",
            ) from exc

        if byte_count == 0:
            raise ContractError("TTS: no PCM bytes received")
        if self.reject_all_silence and peak <= 1e-8:
            raise ServiceError(
                self.service_name,
                "Upstream returned all-silence audio",
            )

        bytes_per_second = (
            self.sample_rate * self.channels * np_dtype.itemsize
        )
        audio_duration_ms = (
            byte_count / float(bytes_per_second)
        ) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000

        yield {
            "type": "done",
            "byte_count": byte_count,
            "peak": peak,
            "ttfb_ms": (
                (first_byte_at - started) * 1000
                if first_byte_at is not None
                else None
            ),
            "total_ms": total_ms,
            "audio_duration_ms": audio_duration_ms,
            "rtf": (
                total_ms / audio_duration_ms
                if audio_duration_ms > 0
                else None
            ),
            "network_chunk_count": network_chunk_count,
            "emitted_chunk_count": emitted_chunk_count,
            "generation_mode": generation_mode,
            "streaming_mode": streaming_mode,
            "upstream_backend": upstream_backend,
            "playback_chunk_ms": target_ms,
        }

    def stream_native_pcm(
        self,
        text: str,
        *,
        voice_index: int,
        trace_id: str | None = None,
        unit_index: int | None = None,
    ) -> Iterator[dict]:
        """Forward upstream PCM with only frame-alignment buffering.

        V10 forwards aligned native PCM without WAV/HLS coalescing. The web media
        plane streams PCM16 continuously and the browser AudioWorklet owns the
        playback ring buffer and underrun clock.
        """
        text = text.strip()
        if not text:
            raise ServiceError(self.service_name, "Text is empty")

        np_dtype = _DTYPE_MAP.get(self.dtype)
        if np_dtype is None:
            raise ContractError(
                f"TTS: unsupported configured dtype {self.dtype!r}"
            )
        frame_bytes = np_dtype.itemsize * self.channels
        pending = bytearray()
        started = time.perf_counter()
        first_byte_at: float | None = None
        byte_count = 0
        peak = 0.0
        emitted = 0
        network_chunks = 0
        streaming_mode = "unknown"
        upstream_backend = "unknown"
        profile = "unknown"
        upstream_commit = "unknown"
        trace_protocol = "unknown"

        def make_audio_event(raw: bytes, *, trace: dict | None = None) -> dict:
            nonlocal byte_count, peak, emitted
            samples = decode_raw_pcm(
                raw,
                dtype=self.dtype,
                channels=self.channels,
            )
            peak = max(peak, float(np.max(np.abs(samples))))
            byte_count += len(raw)
            emitted += 1
            frames = samples.shape[0] if samples.ndim > 1 else samples.size
            return {
                "type": "audio",
                "raw": raw,
                "duration_ms": frames / float(self.sample_rate) * 1000.0,
                "frame_count": int(frames),
                "chunk_index": emitted,
                "ttfb_ms": (first_byte_at - started) * 1000.0,
                "streaming_mode": streaming_mode,
                "upstream_backend": upstream_backend,
                "profile": profile,
                "upstream_commit": upstream_commit,
                "network_chunk_count": network_chunks,
                "trace": trace or {},
            }

        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/api/tts",
                json={
                    "text": text,
                    "voice_index": int(voice_index),
                    "pre_chunked": True,
                    "generation_mode": "stream",
                    "trace_framed": True,
                    "trace_id": trace_id,
                    "unit_index": unit_index,
                },
            ) as r:
                if r.status_code != 200:
                    r.read()
                    raise response_error(self.service_name, r)

                content_type = (r.headers.get("content-type") or "").lower()
                if "application/octet-stream" not in content_type:
                    raise ContractError(
                        "TTS: expected application/octet-stream, got "
                        f"{content_type or '<missing>'}"
                    )

                streaming_mode = r.headers.get("X-Audio-Streaming") or "unknown"
                upstream_backend = r.headers.get("X-TTS-Backend") or "unknown"
                profile = r.headers.get("X-TTS-Profile") or "unknown"
                upstream_commit = r.headers.get("X-VieNeu-Upstream-Commit") or "unknown"
                trace_protocol = r.headers.get("X-TTS-Trace-Protocol") or "none"
                if profile != "realtime":
                    raise ContractError(
                        "TTS realtime contract requires realtime profile, got "
                        f"{profile!r}"
                    )
                if streaming_mode != "frame-pcm":
                    raise ContractError(
                        "TTS realtime contract requires frame-pcm, got "
                        f"{streaming_mode!r}"
                    )
                if "onnx" not in upstream_backend.lower():
                    raise ContractError(
                        "TTS realtime contract requires ONNX backend, got "
                        f"{upstream_backend!r}"
                    )
                if trace_protocol != "tts1-f32":
                    raise ContractError(
                        "TTS realtime trace contract requires tts1-f32, got "
                        f"{trace_protocol!r}"
                    )

                network_iter = (
                    iter((r.content,)) if r.is_stream_consumed else r.iter_raw()
                )
                for network_chunk in network_iter:
                    if not network_chunk:
                        continue
                    network_chunks += 1
                    now = time.perf_counter()
                    if first_byte_at is None:
                        first_byte_at = now
                    pending.extend(network_chunk)

                    # Parse model-yield frames independent of arbitrary HTTP
                    # read boundaries.  A single network read may contain part
                    # of one frame or several complete model frames.
                    while len(pending) >= _TRACE_HEADER.size:
                        magic, sequence, frames, payload_bytes, t0_ns, t1_ns = (
                            _TRACE_HEADER.unpack_from(pending)
                        )
                        if magic != _TRACE_MAGIC:
                            raise ContractError("TTS: invalid trace frame magic")
                        expected_bytes = int(frames) * frame_bytes
                        if payload_bytes != expected_bytes:
                            raise ContractError(
                                "TTS: trace frame size mismatch "
                                f"(declared={payload_bytes}, expected={expected_bytes})"
                            )
                        frame_end = _TRACE_HEADER.size + payload_bytes
                        if len(pending) < frame_end:
                            break
                        raw = bytes(pending[_TRACE_HEADER.size:frame_end])
                        del pending[:frame_end]
                        t2_ns = time.time_ns()
                        yield make_audio_event(raw, trace={
                            "trace_id": trace_id,
                            "unit_index": unit_index,
                            "service_chunk_index": int(sequence),
                            "t0_model_yield_ns": int(t0_ns),
                            "t1_service_yield_ns": int(t1_ns),
                            "t2_client_receive_ns": t2_ns,
                            "t0_t1_ms": (int(t1_ns) - int(t0_ns)) / 1_000_000.0,
                            "t1_t2_ms": (t2_ns - int(t1_ns)) / 1_000_000.0,
                        })

                if pending:
                    raise ContractError(
                        "TTS: trailing non-frame-aligned bytes: "
                        f"{len(pending)}"
                    )
        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            raise ServiceError(self.service_name, "Synthesis timed out") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(self.service_name, f"HTTP error: {exc}") from exc

        if byte_count == 0:
            raise ContractError("TTS: no PCM bytes received")
        if self.reject_all_silence and peak <= 1e-8:
            raise ServiceError(self.service_name, "Upstream returned all-silence audio")

        bytes_per_second = self.sample_rate * self.channels * np_dtype.itemsize
        audio_duration_ms = byte_count / float(bytes_per_second) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        yield {
            "type": "done",
            "byte_count": byte_count,
            "peak": peak,
            "ttfb_ms": (
                (first_byte_at - started) * 1000.0
                if first_byte_at is not None
                else None
            ),
            "total_ms": total_ms,
            "audio_duration_ms": audio_duration_ms,
            "rtf": total_ms / audio_duration_ms if audio_duration_ms > 0 else None,
            "network_chunk_count": network_chunks,
            "emitted_chunk_count": emitted,
            "generation_mode": "stream",
            "streaming_mode": streaming_mode,
            "upstream_backend": upstream_backend,
            "profile": profile,
            "upstream_commit": upstream_commit,
            "trace_protocol": trace_protocol,
        }

    def synthesize_sentence_pcm(
        self,
        text: str,
        *,
        voice_index: int,
        generation_mode: str | None = None,
    ) -> dict:
        """Synthesize one complete semantic sentence and return raw PCM.

        This method is retained for the V6 sentence-buffered compatibility
        path. V7 normally uses ``stream_pcm_chunks`` so native PCM can be
        forwarded while the current sentence is still being synthesized.
        """
        text = text.strip()
        if not text:
            raise ServiceError(self.service_name, "Text is empty")

        mode = (generation_mode or self.generation_mode).strip().lower()
        if mode not in {"sentence", "stream", "auto"}:
            raise ValueError("generation_mode must be sentence, stream, or auto")

        started = time.perf_counter()
        first_byte_at: float | None = None
        parts: list[bytes] = []
        network_chunk_count = 0
        streaming_mode = "unknown"
        resolved_mode = mode
        upstream_backend = "unknown"

        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/api/tts",
                json={
                    "text": text,
                    "voice_index": int(voice_index),
                    "pre_chunked": True,
                    "generation_mode": mode,
                },
            ) as r:
                if r.status_code != 200:
                    r.read()
                    raise response_error(self.service_name, r)

                content_type = (r.headers.get("content-type") or "").lower()
                if "application/octet-stream" not in content_type:
                    raise ContractError(
                        "TTS: expected application/octet-stream, got "
                        f"{content_type or '<missing>'}"
                    )

                streaming_mode = r.headers.get("X-Audio-Streaming") or "unknown"
                resolved_mode = r.headers.get("X-TTS-Generation-Mode") or mode
                upstream_backend = r.headers.get("X-TTS-Backend") or "unknown"

                if r.is_stream_consumed:
                    network_iter = iter((r.content,))
                else:
                    network_iter = r.iter_raw()

                for part in network_iter:
                    if not part:
                        continue
                    network_chunk_count += 1
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()
                    parts.append(part)
        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            raise ServiceError(self.service_name, "Synthesis timed out") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(self.service_name, f"HTTP error: {exc}") from exc

        raw = b"".join(parts)
        samples = decode_raw_pcm(raw, dtype=self.dtype, channels=self.channels)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if self.reject_all_silence and peak <= 1e-8:
            raise ServiceError(self.service_name, "Upstream returned all-silence audio")

        frames = samples.shape[0] if samples.ndim > 1 else samples.size
        audio_duration_ms = frames / float(self.sample_rate) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0

        return {
            "raw": raw,
            "byte_count": len(raw),
            "peak": peak,
            "ttfb_ms": (
                (first_byte_at - started) * 1000.0
                if first_byte_at is not None
                else None
            ),
            "total_ms": total_ms,
            "audio_duration_ms": audio_duration_ms,
            "rtf": total_ms / audio_duration_ms if audio_duration_ms > 0 else None,
            "network_chunk_count": network_chunk_count,
            "generation_mode": resolved_mode,
            "streaming_mode": streaming_mode,
            "upstream_backend": upstream_backend,
        }

    def save_pcm_wav(self, raw: bytes, *, prefix: str = "answer") -> str:
        """Save aligned PCM as a PCM16 replay WAV and return its path."""
        # Validate before writing, then use the same PCM16 wrapper as live
        # Gradio playback so replay and realtime output have identical sample
        # rate/channel semantics.
        decode_raw_pcm(raw, dtype=self.dtype, channels=self.channels)
        self._prune_outputs()
        out = self.output_dir / f"tts-{prefix}-{uuid.uuid4().hex}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(
            pcm_to_wav_bytes(
                raw,
                dtype=self.dtype,
                channels=self.channels,
                sample_rate=self.sample_rate,
            )
        )
        return str(out)

    def stream_pcm_chunks(
        self,
        text: str,
        *,
        voice_index: int,
    ) -> Iterator[dict]:
        """Yield low-latency PCM chunks without wrapping every chunk as WAV.

        The orchestrator uses this path in V4 and performs one WAV wrapping
        step only after PCM has been coalesced into browser-sized playout
        chunks. That avoids hundreds of tiny WAV/FFmpeg conversions in Gradio.
        """
        yield from self._raw_stream(
            text,
            voice_index=voice_index,
            include_wav=False,
        )

    def stream_wav_chunks(
        self,
        text: str,
        *,
        voice_index: int,
    ) -> Iterator[dict]:
        """Compatibility path yielding standalone WAV chunks."""
        yield from self._raw_stream(
            text,
            voice_index=voice_index,
            include_wav=True,
        )

    def synthesize(
        self,
        text: str,
        *,
        voice_index: int,
    ) -> TTSResult:
        started = time.perf_counter()
        raw_chunks: list[bytes] = []
        first_ttfb: float | None = None
        done: dict | None = None

        for event in self._raw_stream(
            text,
            voice_index=voice_index,
            include_wav=False,
        ):
            if event["type"] == "audio":
                raw_chunks.append(event["raw"])
                if first_ttfb is None:
                    first_ttfb = event.get("ttfb_ms")
            else:
                done = event

        raw = b"".join(raw_chunks)
        samples = decode_raw_pcm(
            raw,
            dtype=self.dtype,
            channels=self.channels,
        )

        self._prune_outputs()
        out = self.output_dir / f"tts-{uuid.uuid4().hex}.wav"
        write_wav(out, self.sample_rate, samples)

        return TTSResult(
            audio_path=str(out),
            sample_rate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            duration_s=(
                samples.shape[0] / float(self.sample_rate)
            ),
            byte_count=len(raw),
            ttfb_ms=first_ttfb,
            total_ms=(done or {}).get(
                "total_ms",
                (time.perf_counter() - started) * 1000,
            ),
        )
