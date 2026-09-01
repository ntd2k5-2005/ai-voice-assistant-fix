from __future__ import annotations

import argparse
import json

from .config import AppSettings
from .orchestrator import VoiceOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Check upstream voice services")
    parser.add_argument(
        "--text",
        help="Optional text query to run RAG -> LLM -> TTS without STT.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Skip TTS for the optional text query.",
    )
    args = parser.parse_args()

    orch = VoiceOrchestrator(AppSettings())
    try:
        health = orch.health()
        print(json.dumps(health, ensure_ascii=False, indent=2))
        if args.text:
            result = orch.process(
                query_text=args.text,
                synthesize=not args.no_tts,
            )
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        orch.close()


if __name__ == "__main__":
    main()
