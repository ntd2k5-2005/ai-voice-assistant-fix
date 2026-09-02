# Release verification status

Nhãn trong tài liệu này có nghĩa:

- `FUNCTIONAL_PASS`: đường code/runtime đã được chạy với fixture hoặc service thật.
- `DEGRADED`: chạy được nhưng metric thực tế chưa đạt target realtime.
- `NOT_EVALUATED`: thiếu corpus, hardware hoặc môi trường phù hợp; không phải PASS.

## FUNCTIONAL_PASS

- Runtime profile/manifest/port và exact model identity; TTS backbone/codec
  revision phải được health và release verifier xác nhận là `enforced`.
- Conversation isolation, lifecycle, completed-only history và reset.
- Text/voice turn, WebSocket control cursor reconnect và PCM cursor reconnect.
- Cancellation/barge-in ownership; stale control/PCM bị loại.
- Browser VAD logic với live PCM capture, pre-roll, adaptive noise floor,
  debounce, min speech và endpoint silence bằng synthetic frames.
- Grounded answer có provenance; factual no-context trả deterministic no-answer.
- Knowledge upload, retrieve, cite và delete.
- Chrome headless boot, frontend JS initialization và runtime health rendering.

## DEGRADED

- VieNeu ONNX có aggregate RTF dưới 1 trên máy kiểm tra nhưng warmup vẫn có
  inter-chunk gap lớn; health chủ động báo `degraded`.
- Synthetic TTS→STT round-trip chạy được nhưng transcript chưa chính xác hoàn toàn.

## NOT_EVALUATED

- Chất lượng STT theo vùng miền/noise/code-switch/tên riêng vì chưa có labelled corpus.
- Retrieval answerability threshold/quality vì chưa có labelled RAG calibration set.
- Barge-in speech-start → speaker-stop trên người thật và thiết bị audio thật.
- Colab T4 runtime E2E trong phiên Windows hiện tại.
- Acceptance data trong `acceptance/` đang trống.
