# Eureka V10 Architecture

V10 tách ba đường độc lập: control, speech scheduling và media.

```text
LLM NDJSON reader
   |-- WebSocket control queue -> browser UI
   `-- unbounded text spool -> speech segmentation worker -> bounded TTS queue
                                                        -> VieNeu ONNX infer_stream
                                                        -> bounded PCM queue
                                                        -> raw HTTP PCM
                                                        -> AudioWorklet ring buffer
```

## Invariants

- TTS chậm không được chặn LLM network reader hoặc UI token stream.
- Chỉ speech segmentation worker chịu bounded TTS queue backpressure.
- Một TTS worker duy nhất để tránh concurrent model inference không được chứng minh an toàn.
- Server không giả lập browser playback clock.
- Browser là nguồn sự thật cho buffered frames, underrun và rebuffer.
- AudioWorklet rebuffer sau underrun; không tạo hàng trăm AudioBufferSourceNode.
- VieNeu realtime runtime pin ONNX/CPU từ upstream lock bằng `uv sync --frozen`.
