# Eureka TTS Service V10

Realtime service dùng VieNeu v3 Turbo ONNX/CPU. Môi trường VieNeu không được cài bằng dependency PyPI của service; setup đọc `runtime.json`, checkout đúng upstream commit và chạy `uv sync --frozen` để tái tạo lock environment.

Service warmup native `infer_stream()` khi khởi động và công khai TTFB/RTF/inter-chunk gap trong `/health/ready`. Capability (`ONNX + frame-pcm`) và performance (`realtime-headroom` hoặc `degraded`) được báo riêng.

API chính: `POST /api/tts`, raw float32 PCM 48 kHz mono.
