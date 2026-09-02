# Eureka acceptance suite

Thư mục này là khung dữ liệu acceptance cho module-level và end-to-end tests.
Repository không chứa ground truth giả. Một nhóm rỗng phải được báo là
`NOT_EVALUATED`, không được tính là pass.

## STT

Mỗi thư mục con chứa audio thật và một file sidecar cùng tên, ví dụ
`sample-001.wav` và `sample-001.json`. Sidecar tối thiểu có:

```json
{
  "sample_id": "sample-001",
  "audio_file": "sample-001.wav",
  "expected_transcript": "transcript do người đánh nhãn cung cấp",
  "speaker_region": "north|central|south|unknown",
  "environment": "clean|fan|classroom|event|other",
  "microphone": "laptop|external|unknown",
  "contains_code_switch": false,
  "contains_huit_terms": false
}
```

Không commit audio nhạy cảm hoặc dữ liệu không có quyền sử dụng.

Các fixture PCM/VAD sinh tổng hợp trong automated tests chỉ chứng minh
`FUNCTIONAL_PASS` cho state/endpoint/cancellation. Chúng không được tính là
`QUALITY_VALIDATED` cho chất lượng nhận dạng, giọng đọc hay giọng vùng miền.

## RAG JSONL

- `answerable.jsonl`: `case_id`, `query`, `expected_answer_facts` và
  `relevant_document_ids`.
- `unanswerable.jsonl`: `case_id`, `query`, `reason_unanswerable`.
- `expected_sources.jsonl`: `case_id`, `expected_document_ids`; có thể bổ sung
  page/section khi ground truth đã được kiểm duyệt.

## Conversation JSONL

- `multi_turn.jsonl`: conversation có danh sách turn và kỳ vọng theo từng turn.
- `barge_in.jsonl`: thời điểm speech-start, deadline dừng loa, và policy history.

## Realtime scenarios

`realtime/scenarios.json` là một JSON array. Mỗi scenario phải khai báo target
latency/continuity cần kiểm tra và runtime profile áp dụng. Không đặt threshold
chất lượng model nếu chưa có baseline thực đo và quyết định của team.
