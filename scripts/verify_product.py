from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import websockets
from scipy.io import wavfile

from runtime_config import load_profile


PCM_HEADER = struct.Struct("<4sHHIIIIQQQQQQ")
ROOT = Path(__file__).resolve().parents[1]


def service_url(profile_name: str, service: str) -> str:
    resolved_name, profile = load_profile(ROOT, profile_name)
    try:
        port = profile["ports"][service]
    except KeyError:
        raise ValueError(f"runtime profile {resolved_name!r} has no {service!r} service") from None
    return f"http://127.0.0.1:{port}"


async def consume_control(base_ws: str, turn_id: str) -> list[dict]:
    events: list[dict] = []
    cursor = 0
    # Deliberately reconnect once to verify cursor replay on the real server.
    async with websockets.connect(f"{base_ws}/ws/turn/{quote(turn_id)}") as socket:
        while True:
            event = json.loads(await socket.recv())
            if event.get("type") != "keepalive":
                events.append(event)
                cursor = int(event.get("event_sequence") or cursor)
                break
    async with websockets.connect(
        f"{base_ws}/ws/turn/{quote(turn_id)}?after={cursor}"
    ) as socket:
        while True:
            event = json.loads(await socket.recv())
            if event.get("type") != "keepalive":
                events.append(event)
                cursor = int(event.get("event_sequence") or cursor)
            if event.get("type") == "stream_closed":
                break
    return events


async def consume_audio(base_ws: str, turn_id: str) -> dict:
    sequences: list[int] = []
    byte_count = 0
    async with websockets.connect(
        f"{base_ws}/ws/turn/{quote(turn_id)}/audio",
        max_size=16 * 1024 * 1024,
    ) as socket:
        while True:
            message = await socket.recv()
            if isinstance(message, str):
                if json.loads(message).get("type") == "end":
                    break
                continue
            if len(message) < PCM_HEADER.size:
                raise RuntimeError("short PCM frame")
            fields = PCM_HEADER.unpack_from(message)
            if fields[0] != b"V10P" or fields[1] != 1:
                raise RuntimeError("invalid PCM protocol")
            sequence, frames = fields[3], fields[6]
            if len(message) - fields[2] != frames * 2:
                raise RuntimeError("misaligned PCM frame")
            sequences.append(sequence)
            byte_count += len(message) - fields[2]
    if sequences and sequences != list(range(1, len(sequences) + 1)):
        raise RuntimeError(f"non-contiguous PCM: {sequences[:10]}")
    return {"chunks": len(sequences), "bytes": byte_count}


async def finish_turn(base_ws: str, info: dict) -> dict:
    control, audio = await asyncio.gather(
        consume_control(base_ws, info["turn_id"]),
        consume_audio(base_ws, info["turn_id"]),
    )
    errors = [event for event in control if event.get("type") == "error"]
    if errors:
        raise RuntimeError(f"turn errors: {errors}")
    done = next((event for event in control if event.get("type") == "done"), None)
    if done is None:
        raise RuntimeError("turn ended without done")
    return {"info": info, "events": control, "audio": audio, "done": done}


async def start_text(client: httpx.AsyncClient, conversation_id: str, query: str) -> dict:
    response = await client.post(
        "/api/turn/text",
        json={
            "conversation_id": conversation_id,
            "query": query,
            "rag_mode": "dense",
            "rerank": False,
            "top_k": 3,
            "voice_index": 2,
        },
    )
    response.raise_for_status()
    return response.json()


async def main_async(
    base_url: str,
    include_voice: bool,
    profile_name: str | None = None,
) -> dict:
    base_url = base_url.rstrip("/")
    base_ws = base_url.replace("https://", "wss://").replace("http://", "ws://")
    timeout = httpx.Timeout(240.0)
    report: dict = {"started_at": time.time(), "base_url": base_url}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        health = (await client.get("/health/ready")).raise_for_status().json()
        if not health.get("ready"):
            raise RuntimeError("gateway is not ready")
        runtime_profile = health["runtime"]["profile"]
        if profile_name is not None and profile_name != runtime_profile:
            raise RuntimeError(
                f"gateway profile mismatch: expected={profile_name} actual={runtime_profile}"
            )
        profile_name = profile_name or runtime_profile
        tts_url = service_url(profile_name, "tts")
        report["profile"] = profile_name
        report["manifest"] = health["runtime"]["model_manifest_id"]
        report["service_urls"] = {"tts": tts_url}

        conversation = (await client.post("/api/conversations")).raise_for_status().json()
        conversation_id = conversation["conversation_id"]

        first_info = await start_text(
            client,
            conversation_id,
            "Mã kiểm tra không tồn tại trong kho tri thức là gì?",
        )
        first = await finish_turn(base_ws, first_info)
        if first["done"]["grounding"]["policy"] != "deterministic-no-answer":
            raise RuntimeError("factual empty-context query did not use no-answer gate")
        report["no_answer"] = {
            "answer": first["done"]["answer"],
            "audio": first["audio"],
            "generation": first_info["generation"],
        }

        document_id = f"product-smoke-{int(time.time())}"
        content = (
            "Tài liệu xác minh sản phẩm Eureka. "
            "Mã xác minh sản phẩm Eureka là HOA-SEN-2026."
        ).encode("utf-8")
        uploaded = await client.post(
            "/api/knowledge/upload",
            files={"file": ("product-smoke.txt", content, "text/plain")},
            data={"document_id": document_id},
        )
        uploaded.raise_for_status()
        try:
            grounded_info = await start_text(
                client,
                conversation_id,
                "Mã xác minh sản phẩm Eureka là gì?",
            )
            grounded = await finish_turn(base_ws, grounded_info)
            sources = grounded["done"].get("sources") or []
            if not any(source.get("document_id") == document_id for source in sources):
                raise RuntimeError(f"expected provenance {document_id!r}, got {sources!r}")
            report["grounded"] = {
                "answer": grounded["done"]["answer"],
                "source_ids": [source.get("document_id") for source in sources],
                "audio": grounded["audio"],
                "generation": grounded_info["generation"],
                "timings": grounded["done"].get("timings"),
            }
        finally:
            deleted = await client.delete(f"/api/knowledge/documents/{quote(document_id)}")
            deleted.raise_for_status()

        history = (await client.get(f"/api/conversations/{conversation_id}")).raise_for_status().json()
        complete = [turn for turn in history["turns"] if turn["state"] == "completed"]
        if len(complete) != 2 or history["generation"] != 2:
            raise RuntimeError("multi-turn history/generation contract failed")
        report["multi_turn"] = {"completed_turns": len(complete), "generation": history["generation"]}

        cancel_info = await start_text(client, conversation_id, "Xin chào")
        cancelled = (
            await client.post(
                f"/api/turn/{cancel_info['turn_id']}/cancel",
                json={"reason": "barge_in"},
            )
        ).raise_for_status().json()
        cancel_audio = await consume_audio(base_ws, cancel_info["turn_id"])
        if cancelled["state"] != "interrupted" or cancel_audio["bytes"] != 0:
            raise RuntimeError("barge-in cancellation leaked audio")
        report["barge_in"] = {
            "state": cancelled["state"],
            "stale_audio_bytes": cancel_audio["bytes"],
        }

        if include_voice:
            tts = await client.post(
                f"{tts_url}/api/tts",
                json={
                    "text": "Xin chào, đây là kiểm tra hội thoại bằng giọng nói.",
                    "voice_index": 2,
                    "pre_chunked": True,
                    "generation_mode": "stream",
                },
            )
            tts.raise_for_status()
            samples = np.frombuffer(tts.content, dtype="<f4")
            with tempfile.TemporaryDirectory(prefix="eureka-product-smoke-") as temp:
                audio_path = Path(temp) / "voice.wav"
                wavfile.write(audio_path, 48_000, samples)
                with audio_path.open("rb") as handle:
                    voice_start = await client.post(
                        "/api/turn/voice",
                        files={"audio": ("voice.wav", handle, "audio/wav")},
                        data={
                            "conversation_id": conversation_id,
                            "rag_mode": "dense",
                            "rerank": "false",
                            "top_k": "3",
                            "voice_index": "2",
                        },
                    )
            voice_start.raise_for_status()
            voice = await finish_turn(base_ws, voice_start.json())
            report["voice"] = {
                "transcript": voice["done"]["transcript"],
                "audio": voice["audio"],
                "timings": voice["done"].get("timings"),
            }

        report["duration_s"] = round(time.time() - report["started_at"], 3)
        report["status"] = "FUNCTIONAL_PASS"
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Real functional product E2E")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--profile",
        default=os.getenv("RUNTIME_PROFILE"),
        help="Expected runtime profile; otherwise resolve it from gateway health",
    )
    parser.add_argument("--include-voice", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(main_async(args.base_url, args.include_voice, args.profile)),
            # Windows PowerShell may still expose a CP1252 stdout even though
            # every HTTP payload is UTF-8. Escaping keeps the verifier portable.
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
