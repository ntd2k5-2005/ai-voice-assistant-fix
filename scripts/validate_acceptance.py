from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ROOT / "acceptance"
JSONL_REQUIREMENTS = {
    ACCEPTANCE / "rag" / "answerable.jsonl": {
        "case_id", "query", "expected_answer_facts", "relevant_document_ids"
    },
    ACCEPTANCE / "rag" / "unanswerable.jsonl": {
        "case_id", "query", "reason_unanswerable"
    },
    ACCEPTANCE / "rag" / "expected_sources.jsonl": {
        "case_id", "expected_document_ids"
    },
    ACCEPTANCE / "conversation" / "multi_turn.jsonl": {
        "case_id", "turns"
    },
    ACCEPTANCE / "conversation" / "barge_in.jsonl": {
        "case_id", "speech_start_ms", "speaker_stop_deadline_ms"
    },
}
STT_GROUPS = ("clean", "noise", "regional", "code_switch", "huit_terms")
STT_REQUIRED = {
    "sample_id", "audio_file", "expected_transcript", "speaker_region",
    "environment", "microphone", "contains_code_switch", "contains_huit_terms",
}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}


def validate_jsonl(path: Path, required: set[str]) -> int:
    count = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
        missing = required - set(value)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
        count += 1
    return count


def validate_stt_group(path: Path) -> int:
    audio_files = [item for item in path.iterdir() if item.suffix.casefold() in AUDIO_SUFFIXES]
    for audio in audio_files:
        sidecar = audio.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"{audio}: missing sidecar {sidecar.name}")
        value = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{sidecar}: sidecar must contain a JSON object")
        missing = STT_REQUIRED - set(value)
        if missing:
            raise ValueError(f"{sidecar}: missing fields {sorted(missing)}")
        if value["audio_file"] != audio.name:
            raise ValueError(f"{sidecar}: audio_file must equal {audio.name!r}")
    return len(audio_files)


def main() -> None:
    total = 0
    for group in STT_GROUPS:
        path = ACCEPTANCE / "stt" / group
        count = validate_stt_group(path)
        total += count
        state = "READY" if count else "NOT_EVALUATED"
        print(f"{state:13} {path.relative_to(ROOT)} cases={count}")
    for path, required in JSONL_REQUIREMENTS.items():
        count = validate_jsonl(path, required)
        total += count
        state = "READY" if count else "NOT_EVALUATED"
        print(f"{state:13} {path.relative_to(ROOT)} cases={count}")

    scenarios_path = ACCEPTANCE / "realtime" / "scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list):
        raise ValueError(f"{scenarios_path} must contain a JSON array")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"{scenarios_path}[{index}] must be an object")
        missing = {"case_id", "runtime_profile", "targets"} - set(scenario)
        if missing:
            raise ValueError(f"{scenarios_path}[{index}] missing fields {sorted(missing)}")
    total += len(scenarios)
    state = "READY" if scenarios else "NOT_EVALUATED"
    print(f"{state:13} {scenarios_path.relative_to(ROOT)} cases={len(scenarios)}")
    print(f"TOTAL_CASES {total}")


if __name__ == "__main__":
    main()
