# API Contracts

## V10 web media plane

- `POST /api/conversations`: tạo conversation mới.
- `GET /api/conversations/{conversation_id}`: đọc lifecycle, history và provenance.
- `DELETE /api/conversations/{conversation_id}`: hủy turn còn chạy và reset conversation.
- `POST /api/turn/text`: tạo lượt text trong `conversation_id`.
- `POST /api/turn/voice`: tạo lượt voice trong `conversation_id`.
- `GET /api/turn/{turn_id}`: đọc trạng thái turn.
- `POST /api/turn/{turn_id}/cancel`: cancellation idempotent; `reason=barge_in` chuyển turn sang `interrupted`.
- `WebSocket /ws/turn/{turn_id}?after=<event_sequence>`: control stream có cursor để reconnect.
- `WebSocket /ws/turn/{turn_id}/audio?after=<chunk_index>`: framed PCM16 có
  sequence, trace T0–T5 và bounded cursor replay khi reconnect.
- `GET /api/turn/{turn_id}/audio`: trả `410 Gone`; raw HTTP đã được retire để không cạnh tranh consumer với WebSocket media plane.
- `POST /api/turn/{turn_id}/browser-metrics`: browser gửi AudioWorklet underrun/rebuffer metrics.
- `GET /api/turn/{turn_id}/replay.wav`: replay sau khi TTS hoàn tất.

Mọi event mang `conversation_id`, `turn_id`, server generation và event sequence.
LLM/control, text spool, speech unit, PCM, event replay và AudioWorklet đều có
giới hạn bộ nhớ. Cancellation token đi qua LLM reader, segmenter và TTS; gateway
là ownership gate cuối cùng, còn browser epoch loại stale packet sau barge-in.

Turn lifecycle:

`created → transcribing/retrieving → thinking → speaking → completed`, hoặc
`failed/cancelled/interrupted`.

## Grounding

Greeting/social dùng `social-bypass`. Factual query không có usable context trả
no-answer deterministic mà không gọi LLM. Khi có context, assistant nhận prompt
grounded và provenance được gắn vào turn. Repository chưa có labelled corpus nên
không đặt retrieval-score threshold cảm tính; calibration vẫn `NOT_EVALUATED`.

## TTS realtime

TTS health tách capability và performance. Realtime contract yêu cầu ONNX + `frame-pcm` + upstream locked environment; warmup suite có thể báo `realtime-headroom` hoặc `degraded` tùy máy.
