from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_llm_stream_endpoint_source_contract():
    text=(ROOT/"services/llm/api/main.py").read_text(encoding="utf-8")
    assert '@app.post("/v1/chat/stream"' in text and "application/x-ndjson" in text

def test_v10_setup_reproduces_locked_vieneu_environment():
    setup=(ROOT/"scripts/setup_colab.sh").read_text(encoding="utf-8")
    runtime=(ROOT/"services/tts/runtime.json").read_text(encoding="utf-8")
    assert "uv sync" in setup and "--frozen" in setup and "UV_PROJECT_ENVIRONMENT" in setup
    assert "36c4b501b0634a8f59805e6b529a058fbd30190b" in runtime

def test_v10_tts_pins_onnx_and_runs_multi_shape_warmup():
    text=(ROOT/"services/tts/api.py").read_text(encoding="utf-8")
    assert 'VIENEU_BACKEND = "onnx"' in text
    assert 'return {"mode": VIENEU_MODE, "backend": "onnx"}' in text
    assert '"profiles": ["short", "medium"]' in text
    assert '"environment_policy": "uv-sync-frozen"' in text
    assert '"decoupled-speech-units-websocket-audioworklet-v10"' in text

def test_v10_media_decouples_llm_from_tts_backpressure():
    text=(ROOT/"app/voice_app/media_v10.py").read_text(encoding="utf-8")
    assert '"pipeline_version": "v10"' in text
    assert '"control_transport": "websocket"' in text
    assert '"llm_reader_blocked_by_tts": False' in text
    assert 'speech_spool: queue.Queue' in text
    assert 'threading.Thread(target=speech_segmenter_worker' in text

def test_v10_browser_uses_websocket_and_rebuffering_audioworklet():
    js=(ROOT/"app/voice_app/web_assets/app.js").read_text(encoding="utf-8")
    worklet=(ROOT/"app/voice_app/web_assets/pcm-player-worklet.js").read_text(encoding="utf-8")
    assert "new WebSocket" in js and "/ws/turn/" in js
    assert "EventSource" not in js
    assert 'pcm-player-v10' in js
    assert 'class FloatRingBuffer' in worklet
    assert 'rebufferFrames' in worklet and 'rebuffering' in worklet
    assert "createBufferSource" not in js

def test_v10_web_control_and_media_routes():
    web=(ROOT/"app/voice_app/web.py").read_text(encoding="utf-8")
    assert '@app.websocket("/ws/turn/{turn_id}")' in web
    assert 'media_type="application/octet-stream"' in web
    assert '"X-Media-Plane": "audioworklet-raw-pcm-v10"' in web
    assert 'original_name = _safe_filename(file.filename' in web
    assert 'resolved_id = (document_id or "").strip() or Path(original_name).stem' in web
    assert 'session.audio_queue.put_nowait(pcm16)' in web

def test_knowledge_management_remains_available():
    web=(ROOT/"app/voice_app/web.py").read_text(encoding="utf-8")
    for route in ('@app.get("/api/knowledge/documents")','@app.post("/api/knowledge/upload")'):
        assert route in web
