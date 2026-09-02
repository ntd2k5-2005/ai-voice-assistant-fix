from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
import uuid
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from voice_app.realtime.speech_units import AdaptiveSpeechUnitChunker  # noqa: E402
from runtime_config import load_profile  # noqa: E402


SAMPLE_RATE = 48_000
TRACE_HEADER = struct.Struct("<4sIIIQQ")
VIDEO_TEXT = (
    "Khoa Công nghệ Thông tin của Đại học Công Thương TP.HCM đào tạo các ngành như "
    "Công nghệ thông tin, An toàn thông tin, Khoa học dữ liệu và Trí tuệ nhân tạo. "
    "Năm 2026, khoa đã công bố tầm nhìn mới với chương trình Tiến sĩ CNTT và mở rộng "
    "ngành AI. Khoa có lãnh đạo là TS. Nguyễn Hồng Vũ làm Trưởng khoa và TS. Nguyễn "
    "Thanh Long làm Phó Trưởng khoa.\n\nKhoa hoạt động tại địa chỉ 140 Lê Trọng Tấn "
    "và cung cấp các dịch vụ hỗ trợ như Trung tâm Công nghệ Thông tin cũng như Phòng "
    "Khoa học công nghệ. Bạn có thể truy cập thông tin chi tiết qua website chính thức "
    "của khoa là fit.huit.edu.vn."
)


def current_units(text: str) -> list[str]:
    chunker = AdaptiveSpeechUnitChunker()
    return chunker.push(text) + chunker.flush()


def strategy_units(text: str, strategy: str) -> list[str]:
    units = current_units(text)
    if strategy == "current":
        return units
    if strategy == "single":
        return [" ".join(text.split())]
    if strategy == "fewer":
        # Preserve the interactive first unit, then amortize request startup by
        # combining adjacent steady units. This remains an experiment until the
        # measurements show that it wins.
        result = units[:1]
        for index in range(1, len(units), 2):
            result.append(" ".join(units[index:index + 2]))
        return result
    raise ValueError(strategy)


def stream_unit(client: httpx.Client, url: str, text: str, voice: int, unit: int):
    started = time.perf_counter()
    pending = bytearray()
    rows = []
    with client.stream(
        "POST",
        f"{url.rstrip('/')}/api/tts",
        json={
            "text": text,
            "voice_index": voice,
            "pre_chunked": True,
            "generation_mode": "stream",
            "trace_framed": True,
            "trace_id": uuid.uuid4().hex,
            "unit_index": unit,
        },
    ) as response:
        response.raise_for_status()
        if response.headers.get("X-TTS-Trace-Protocol") != "tts1-f32":
            raise RuntimeError("TTS service does not support tts1-f32 tracing")
        for network_chunk in response.iter_raw():
            received = time.perf_counter()
            pending.extend(network_chunk)
            while len(pending) >= TRACE_HEADER.size:
                magic, sequence, frames, payload_bytes, t0_ns, t1_ns = TRACE_HEADER.unpack_from(pending)
                if magic != b"TTS1" or payload_bytes != frames * 4:
                    raise RuntimeError("invalid TTS1 frame")
                frame_end = TRACE_HEADER.size + payload_bytes
                if len(pending) < frame_end:
                    break
                del pending[:frame_end]
                rows.append({
                    "sequence": sequence,
                    "frames": frames,
                    "audio_ms": frames / SAMPLE_RATE * 1000.0,
                    "received_at": received,
                    "model_yield_ns": t0_ns,
                    "service_yield_ns": t1_ns,
                    "service_pack_ms": (t1_ns - t0_ns) / 1_000_000.0,
                })
    if pending or not rows:
        raise RuntimeError(f"incomplete TTS trace stream ({len(pending)} trailing bytes)")
    finished = time.perf_counter()
    audio_ms = sum(row["audio_ms"] for row in rows)
    gaps = [
        (right["received_at"] - left["received_at"]) * 1000.0
        for left, right in zip(rows, rows[1:])
    ]
    return rows, {
        "unit": unit,
        "chars": len(text),
        "words": len(text.split()),
        "ttfb_ms": (rows[0]["received_at"] - started) * 1000.0,
        "total_ms": (finished - started) * 1000.0,
        "audio_duration_ms": audio_ms,
        "rtf": (finished - started) * 1000.0 / audio_ms,
        "chunk_count": len(rows),
        "max_inter_yield_gap_ms": max(gaps) if gaps else 0.0,
        "p95_inter_yield_gap_ms": (
            statistics.quantiles(gaps, n=20, method="inclusive")[18]
            if len(gaps) > 1 else (gaps[0] if gaps else 0.0)
        ),
    }


def run_strategy(url: str, text: str, voice: int, strategy: str) -> dict:
    units = strategy_units(text, strategy)
    started = time.perf_counter()
    cumulative_audio_ms = 0.0
    lead_timeline = []
    unit_rows = []
    with httpx.Client(timeout=httpx.Timeout(420.0)) as client:
        for unit_index, unit_text in enumerate(units, 1):
            chunks, metric = stream_unit(client, url, unit_text, voice, unit_index)
            unit_rows.append(metric | {"text": unit_text})
            for chunk in chunks:
                cumulative_audio_ms += chunk["audio_ms"]
                elapsed_ms = (chunk["received_at"] - started) * 1000.0
                lead_timeline.append({
                    "unit": unit_index,
                    "chunk": chunk["sequence"],
                    "elapsed_ms": elapsed_ms,
                    "cumulative_audio_ms": cumulative_audio_ms,
                    "audio_lead_ms": cumulative_audio_ms - elapsed_ms,
                })
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "strategy": strategy,
        "request_count": len(units),
        "total_ms": total_ms,
        "audio_duration_ms": cumulative_audio_ms,
        "rtf": total_ms / cumulative_audio_ms,
        "final_audio_lead_ms": cumulative_audio_ms - total_ms,
        "min_audio_lead_ms": min(row["audio_lead_ms"] for row in lead_timeline),
        "realtime_headroom": cumulative_audio_ms > total_ms,
        "units": unit_rows,
        "lead_timeline": lead_timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--voice", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--text", default=VIDEO_TEXT)
    parser.add_argument(
        "--strategies", nargs="+", choices=("current", "fewer", "single"),
        default=("current", "fewer", "single"),
    )
    args = parser.parse_args()
    if args.url is None:
        _, profile = load_profile(ROOT, args.profile)
        args.url = f"http://127.0.0.1:{profile['ports']['tts']}"
    results = []
    for run in range(1, args.runs + 1):
        for strategy in args.strategies:
            row = run_strategy(args.url, args.text, args.voice, strategy)
            row["run"] = run
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
