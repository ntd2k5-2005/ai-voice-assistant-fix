# API Contracts

## V10 web media plane

- `POST /api/turn/text`: tạo lượt text.
- `POST /api/turn/voice`: tạo lượt voice.
- `WebSocket /ws/turn/{turn_id}`: control stream realtime (`stage`, `llm_delta`, speech/TTS metrics, `done`).
- `GET /api/turn/{turn_id}/audio`: một raw PCM16 HTTP stream liên tục.
- `POST /api/turn/{turn_id}/browser-metrics`: browser gửi AudioWorklet underrun/rebuffer metrics.
- `GET /api/turn/{turn_id}/replay.wav`: replay sau khi TTS hoàn tất.

LLM control path và TTS backpressure được tách bằng text spool + speech segmentation worker. WebSocket không đi qua bounded TTS queue.

## TTS realtime

TTS health tách capability và performance. Realtime contract yêu cầu ONNX + `frame-pcm` + upstream locked environment; warmup suite có thể báo `realtime-headroom` hoặc `degraded` tùy máy.
