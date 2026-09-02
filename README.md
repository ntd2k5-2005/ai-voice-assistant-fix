# AI Voice Assistant — Eureka V10

Pipeline chính:

```text
Microphone / text
      -> PTT or VAD (pre-roll + endpoint + barge-in)
      -> conversation / turn ownership
      -> STT
      -> RAG
      -> LLM token stream
         -> WebSocket control/UI (không bị TTS backpressure)
         -> text spool
             -> adaptive semantic speech units
             -> VieNeu v3 Turbo ONNX/CPU infer_stream
                (upstream env reproduced by uv sync --frozen)
             -> framed PCM16 WebSocket
             -> Browser AudioWorklet rebuffering ring buffer
             -> speaker
```

UI chính hỗ trợ chat nhiều lượt, reset conversation, PTT/VAD selector, ngắt lời,
realtime text, provenance, quản lý tài liệu và runtime health. Khi VAD phát hiện
người dùng nói lúc assistant đang phát, browser flush AudioWorklet trước, hủy
turn cũ, loại stale PCM theo epoch rồi gửi câu mới.

V10 giữ các phần đã chứng minh đúng ở V9 (locked VieNeu runtime, semantic speech units, AudioWorklet) và sửa hai vấn đề đo được: TTS queue không còn chặn LLM/control stream, và control plane chuyển từ SSE sang WebSocket để tránh proxy buffering. AudioWorklet dùng ring buffer thật và rebuffer sau underrun để tránh chuỗi micro-stutter.

## VieNeu realtime runtime

Nguồn runtime được pin trong `services/tts/runtime.json`. `scripts/setup_colab.sh` clone đúng commit và chạy `uv sync --frozen` vào `.venvs/tts`. TTS boot warmup cả short/medium input shape và công khai TTFB/RTF/gap trong `/health/ready`. Warmup chậm vẫn được báo `degraded`; UI vẫn mở để demo và chẩn đoán.

## Chạy trên Windows

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.venvs\app\Scripts\python.exe scripts/start_services.py --profile local-balanced --with-web
```

Mở `http://127.0.0.1:7860`. Các profile native Windows dùng ports
`8100`–`8103` để tránh xung đột với service local phổ biến trên `8000`.

## Chạy trên Colab / Linux

```bash
bash scripts/setup_colab.sh
.venvs/app/bin/python scripts/start_services.py --profile colab-t4 --with-web
```

Ports: STT `8000`, RAG `8001`, LLM `8002`, TTS `8003`, web `7860`.

## Realtime contract

Control events đi qua `WebSocket /ws/turn/{turn_id}`; PCM đi riêng qua
`WebSocket /ws/turn/{turn_id}/audio` với frame header T0–T5. LLM reader ghi
control event và text spool mà không bao giờ `put()` trực tiếp vào bounded TTS
queue. Speech segmentation worker mới chịu TTS backpressure.

Browser dùng AudioWorklet ring buffer với initial prebuffer và rebuffer threshold riêng. Sau underrun, worklet đợi đủ cushion rồi mới tiếp tục phát thay vì giật thành nhiều đoạn ngắn.

## Kho tri thức

Upload giữ nguyên filename người dùng và mặc định `document_id` theo tên file gốc; temp path không lộ ra metadata.

## Chẩn đoán

```bash
.venvs/app/bin/python scripts/test_project.py
.venvs/app/bin/python scripts/verify_services.py --profile local-balanced
.venvs/app/bin/python scripts/verify_product.py --profile local-balanced --include-voice
.venvs/app/bin/python scripts/benchmark_tts_v10.py --profile local-balanced
```

Benchmark là công cụ chẩn đoán, không phải điều kiện để mở UI demo.
`verify_product.py` là functional E2E thật: conversation, no-answer, upload,
provenance, LLM/TTS, cancellation stale-audio và voice round-trip.

## Model manifest và runtime profiles

Baseline model nằm trong `config/model_manifest.json`. Device placement và ports
nằm trong `config/runtime_profiles.json`; các profile hiện có là
`local-low-vram`, `local-cpu`, `local-balanced`, `colab-t4` và `server-gpu`.
Không profile nào dùng device `auto`.

Launcher đặt Hugging Face cache tại `.model_cache/huggingface`, dữ liệu RAG tại
`.runtime_data/<profile>/rag`, và tắt telemetry của Hugging Face. Sau khi model
đã được chuẩn bị trong cache, có thể ép cache-only runtime:

```powershell
$env:RUNTIME_OFFLINE = "true"
.venvs\app\Scripts\python.exe scripts/start_services.py --profile local-balanced --with-web
```

`num_gpu=0` ép Ollama chạy CPU; `num_gpu=-1` yêu cầu Ollama tự offload số layer
phù hợp với VRAM. Placement thực tế vẫn phải được kiểm tra bằng health snapshot
và `ollama ps` trong acceptance run.

## Acceptance skeleton

Khung dữ liệu nằm trong `acceptance/`. Các file hiện rỗng có trạng thái
`NOT_EVALUATED`, không được tính là pass:

```bash
python scripts/validate_acceptance.py
```

Nguồn gốc các module xem `PROVENANCE.md`.
