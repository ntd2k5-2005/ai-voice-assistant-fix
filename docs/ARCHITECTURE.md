# Eureka V10 Architecture

V10 tách ba đường độc lập: control, speech scheduling và media.

```text
ConversationHistory (in-memory; replaceable by SQLite/Redis)
   -> Turn state machine + generation ownership
      -> LLM NDJSON reader
   |     |-- bounded replayable WebSocket control -> browser chat
   |     `-- bounded token-capacity spool -> speech segmentation -> bounded TTS queue
                                                        -> VieNeu ONNX infer_stream
                                                        -> bounded PCM queue
                                                        -> framed WebSocket PCM
                                                        -> epoch-owned bounded AudioWorklet ring buffer

Browser mic AudioWorklet -> PTT or VAD (pre-roll/noise gate/endpoint)
                         -> speech start -> flush playout -> cancel old turn
                         -> WAV upload -> new owned turn
```

## Invariants

- TTS chậm không được chặn LLM network reader hoặc UI token stream.
- Chỉ speech segmentation worker chịu bounded TTS queue backpressure.
- Một TTS worker duy nhất để tránh concurrent model inference không được chứng minh an toàn.
- Server không giả lập browser playback clock.
- Browser là nguồn sự thật cho buffered frames, underrun và rebuffer.
- AudioWorklet rebuffer sau underrun; không tạo hàng trăm AudioBufferSourceNode.
- VieNeu realtime runtime pin ONNX/CPU từ upstream lock bằng `uv sync --frozen`.
- Factual no-context không đi qua LLM; social interaction không bị ép RAG refusal.
- Chỉ completed turn đi vào multi-turn prompt; interrupted/failed turn không làm bẩn history.
