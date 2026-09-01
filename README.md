# AI Voice Assistant — Eureka V10

Pipeline chính:

```text
Microphone / text
      -> STT
      -> RAG
      -> LLM token stream
         -> WebSocket control/UI (không bị TTS backpressure)
         -> text spool
             -> adaptive semantic speech units
             -> VieNeu v3 Turbo ONNX/CPU infer_stream
                (upstream env reproduced by uv sync --frozen)
             -> one raw PCM16 HTTP stream
             -> Browser AudioWorklet rebuffering ring buffer
             -> speaker
```

V10 giữ các phần đã chứng minh đúng ở V9 (locked VieNeu runtime, semantic speech units, AudioWorklet) và sửa hai vấn đề đo được: TTS queue không còn chặn LLM/control stream, và control plane chuyển từ SSE sang WebSocket để tránh proxy buffering. AudioWorklet dùng ring buffer thật và rebuffer sau underrun để tránh chuỗi micro-stutter.

## VieNeu realtime runtime

Nguồn runtime được pin trong `services/tts/runtime.json`. `scripts/setup_colab.sh` clone đúng commit và chạy `uv sync --frozen` vào `.venvs/tts`. TTS boot warmup cả short/medium input shape và công khai TTFB/RTF/gap trong `/health/ready`. Warmup chậm vẫn được báo `degraded`; UI vẫn mở để demo và chẩn đoán.

## Chạy trên Colab / Linux

```bash
bash scripts/setup_colab.sh
.venvs/app/bin/python scripts/start_services.py
```

Sau đó mở terminal/cell khác:

```bash
cd app
../.venvs/app/bin/python -m voice_app.web
```

Ports: STT `8000`, RAG `8001`, LLM `8002`, TTS `8003`, web `7860`.

## Realtime contract

Control events đi qua `WebSocket /ws/turn/{turn_id}`; PCM đi riêng qua `GET /api/turn/{turn_id}/audio`. LLM reader ghi control event và text spool mà không bao giờ `put()` trực tiếp vào bounded TTS queue. Speech segmentation worker mới chịu TTS backpressure.

Browser dùng AudioWorklet ring buffer với initial prebuffer và rebuffer threshold riêng. Sau underrun, worklet đợi đủ cushion rồi mới tiếp tục phát thay vì giật thành nhiều đoạn ngắn.

## Kho tri thức

Upload giữ nguyên filename người dùng và mặc định `document_id` theo tên file gốc; temp path không lộ ra metadata.

## Chẩn đoán

```bash
.venvs/app/bin/python scripts/test_project.py
.venvs/app/bin/python scripts/benchmark_tts_v10.py
```

Benchmark là công cụ chẩn đoán, không phải điều kiện để mở UI demo.

Nguồn gốc các module xem `PROVENANCE.md`.
