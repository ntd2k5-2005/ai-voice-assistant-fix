from __future__ import annotations

import json
import urllib.request


ENDPOINTS = {
    "stt": "http://127.0.0.1:8000/health/ready",
    "rag": "http://127.0.0.1:8001/health/ready",
    "llm": "http://127.0.0.1:8002/health/ready",
    "tts": "http://127.0.0.1:8003/health/ready",
}


def main() -> None:
    failed = False
    payloads = {}
    for name, url in ENDPOINTS.items():
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
        print("TTS V8 FAIL: expected profile=realtime, backend=onnx, streaming_mode=frame-pcm")
    else:
        print("TTS V8 realtime contract PASS")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
