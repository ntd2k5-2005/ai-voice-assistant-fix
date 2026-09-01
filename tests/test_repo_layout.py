from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_modules_exist():
    expected = [
        ROOT / "services" / "stt" / "app.py",
        ROOT / "services" / "rag" / "rag" / "api" / "app.py",
        ROOT / "services" / "llm" / "api" / "main.py",
        ROOT / "services" / "tts" / "api.py",
        ROOT / "app" / "voice_app" / "orchestrator.py",
        ROOT / "app" / "voice_app" / "media_v10.py",
        ROOT / "app" / "voice_app" / "web_assets" / "pcm-player-worklet.js",
    ]
    assert all(path.exists() for path in expected)


def test_no_nested_llm_model_copy():
    assert not (ROOT / "services" / "llm" / "LLM-Model").exists()


def test_no_voice_integration_double_folder():
    assert not (ROOT / "app" / "voice-integration-app").exists()
