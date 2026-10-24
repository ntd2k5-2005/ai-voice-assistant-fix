# Vietnamese Realtime Voice Knowledge Assistant

> **Nghiên cứu xây dựng và đánh giá hệ thống AI đàm thoại tiếng Việt trên môi trường cục bộ**

Hệ thống AI đàm thoại tiếng Việt theo kiến trúc mô-đun, tích hợp **Speech-to-Text (STT)**, **Retrieval-Augmented Generation (RAG)**, **Large Language Model (LLM)** và **Text-to-Speech (TTS)** trong một pipeline hội thoại thống nhất.

Hệ thống hỗ trợ hội thoại nhiều lượt, Push-to-Talk, Voice Activity Detection, ngắt lời khi trợ lý đang phát âm thanh, quản lý kho tri thức và truy xuất có nguồn tham chiếu.

Trọng tâm của dự án không phải đề xuất một kiến trúc học sâu mới, mà là **thiết kế, tích hợp, tối ưu và đánh giá một hệ thống AI đàm thoại tiếng Việt có khả năng vận hành và tái lập trong môi trường cục bộ**.

---

## 1. Mục tiêu

Dự án hướng đến một trợ lý giọng nói có thể:

- nhận đầu vào bằng **văn bản hoặc giọng nói tiếng Việt**;
- chuyển giọng nói thành văn bản bằng STT;
- truy xuất tri thức nội bộ bằng RAG;
- sinh phản hồi bằng LLM theo ngữ cảnh hội thoại;
- tổng hợp câu trả lời thành âm thanh và phát theo luồng;
- hỗ trợ hội thoại nhiều lượt;
- hỗ trợ **Push-to-Talk (PTT)** và **Voice Activity Detection (VAD)**;
- hỗ trợ **barge-in**, cho phép người dùng ngắt lời khi trợ lý đang nói;
- hiển thị nguồn tri thức được sử dụng trong câu trả lời;
- cho phép nạp, quản lý và xóa tài liệu trong kho tri thức;
- vận hành mà **không phụ thuộc vào API suy luận AI bên ngoài trong mỗi lượt hội thoại**.

Trong dự án này, *triển khai cục bộ* được hiểu là các mô hình STT, RAG, LLM và TTS được vận hành trong cùng môi trường thực thi của hệ thống, thay vì gửi nội dung hội thoại đến một dịch vụ AI inference bên ngoài.

Google Colab T4 được sử dụng như một môi trường thực nghiệm có cấu hình phần cứng xác định để benchmark và kiểm tra khả năng tái lập. Đây không phải môi trường triển khai đích và kiến trúc không phụ thuộc riêng vào Colab.

---

## 2. Tính năng chính

### 2.1. Hội thoại

- Text chat.
- Voice chat.
- Multi-turn conversation.
- Tạo cuộc trò chuyện mới.
- Reset conversation.
- Quản lý trạng thái từng lượt hội thoại.
- Chỉ các lượt hoàn tất mới được đưa vào lịch sử.
- Lượt bị hủy, lỗi hoặc bị ngắt không làm nhiễu context của lượt sau.

### 2.2. Thu nhận giọng nói

- **Push-to-Talk**: người dùng chủ động bắt đầu và kết thúc lượt nói.
- **VAD**: tự phát hiện bắt đầu và kết thúc tiếng nói.
- Pre-roll để hạn chế mất phần đầu câu.
- Adaptive noise floor.
- Endpoint silence.
- Barge-in khi người dùng bắt đầu nói trong lúc trợ lý đang phát âm thanh.

### 2.3. RAG và kho tri thức

- Upload tài liệu.
- Liệt kê tài liệu.
- Xóa tài liệu.
- Dense retrieval.
- Sparse retrieval với BM25/SQLite FTS5.
- Hybrid retrieval.
- Reciprocal Rank Fusion.
- Reranking tùy chọn.
- Provenance: câu trả lời có thể kèm nguồn tài liệu.
- Quản lý vòng đời và phiên bản tài liệu.
- **Deterministic no-answer** cho truy vấn factual khi không có đủ ngữ cảnh phù hợp.

### 2.4. Streaming và phát âm thanh

- LLM stream token.
- TTS sinh âm thanh theo luồng.
- Control plane và media plane tách riêng.
- PCM16 được truyền qua WebSocket riêng.
- Browser phát audio bằng **AudioWorklet ring buffer**.
- Prebuffer và rebuffer giúp giảm giật âm thanh khi tốc độ sinh PCM không đều.
- Stale audio/control event của lượt cũ bị loại theo turn ownership và playback epoch.

### 2.5. Khả năng tái lập

- Model manifest tập trung.
- Runtime profile tập trung.
- Pin revision/checkpoint cho các model quan trọng.
- Kiểm tra model identity qua health endpoint.
- Cấu hình riêng cho Windows, Colab T4 và server GPU.
- Hỗ trợ cache-only/offline runtime sau khi model đã được tải đầy đủ.

---

## 3. Kiến trúc tổng thể

```text
                         ┌─────────────────────────┐
                         │       Browser UI        │
                         │ Text / PTT / VAD / KB   │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │        FastAPI Web Gateway        │
                    │ conversation / turn ownership     │
                    │ cancellation / grounding policy   │
                    └─────────────────┬─────────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                  │                  │
                   ▼                  ▼                  ▼
              ┌─────────┐        ┌─────────┐        ┌─────────┐
Voice input → │   STT   │ text → │   RAG   │context→│   LLM   │
              │Whisper  │        │Dense +  │        │ stream  │
              └─────────┘        │ Sparse  │        └────┬────┘
                                 └─────────┘             │
                                                        │ token stream
                                                        ▼
                                              ┌────────────────────┐
                                              │ Speech segmentation │
                                              │ + text spool        │
                                              └─────────┬──────────┘
                                                        │
                                                        ▼
                                                ┌─────────────┐
                                                │     TTS     │
                                                │ VieNeu ONNX │
                                                └──────┬──────┘
                                                       │ PCM16
                                                       ▼
                                            ┌─────────────────────┐
                                            │ WebSocket media path │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │ Browser AudioWorklet │
                                            │ ring buffer/rebuffer │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                                    Speaker
```

Kiến trúc được tách thành ba luồng logic:

### Control path

Chuyển các sự kiện:

- trạng thái turn;
- token LLM;
- grounding;
- provenance;
- lỗi;
- hoàn tất lượt;
- trạng thái giao diện.

### Speech scheduling path

Nhận text stream từ LLM, tổ chức thành các speech unit và lập lịch cho TTS.

### Media path

Nhận PCM từ TTS và truyền độc lập tới trình duyệt.

Việc tách ba luồng giúp độ trễ của TTS không trực tiếp chặn luồng token của LLM hoặc việc cập nhật UI.

Chi tiết kiến trúc xem tại:

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 4. Vòng đời một lượt hội thoại

Một turn có vòng đời điển hình:

```text
created
  ↓
transcribing
  ↓
retrieving
  ↓
thinking
  ↓
speaking
  ↓
completed
```

Các trạng thái kết thúc khác:

```text
failed
cancelled
interrupted
```

Các nguyên tắc chính:

- mỗi turn có ownership riêng;
- chỉ generation hiện tại được phép cập nhật conversation;
- audio của turn cũ không được phát sau khi turn mới đã chiếm quyền;
- khi barge-in xảy ra, browser flush playback trước khi tiếp tục turn mới;
- stale control event và stale PCM bị loại;
- chỉ turn `completed` mới được đưa vào history của LLM.

---

## 5. Các mô hình và thành phần AI

Model identity được khai báo tại:

[`config/model_manifest.json`](config/model_manifest.json)

| Thành phần | Model / kỹ thuật | Runtime |
|---|---|---|
| STT | `vinai/PhoWhisper-base` | Transformers / PyTorch |
| VAD | Silero VAD 6.2.1 | PyTorch |
| Dense embedding | `intfloat/multilingual-e5-small` | SentenceTransformers |
| Sparse retrieval | BM25 / SQLite FTS5 | Local |
| Fusion | Reciprocal Rank Fusion | Local |
| Reranker | `Alibaba-NLP/gte-multilingual-reranker-base` | Transformers |
| LLM | `qwen3.5:4b` | Ollama |
| TTS | VieNeu-TTS v3 Turbo | ONNX |
| Browser audio | PCM16 + AudioWorklet | Web Audio API |

### Model revisions

Các checkpoint quan trọng được pin để tăng khả năng tái lập:

**PhoWhisper-base**

```text
7ebdb9e88f5cc5271fb88f4d642c82ff9388650e
```

**multilingual-e5-small**

```text
614241f622f53c4eeff9890bdc4f31cfecc418b3
```

**GTE multilingual reranker weights**

```text
8215cf04918ba6f7b6a62bb44238ce2953d8831c
```

**GTE remote code revision**

```text
40ced75c3017eb27626c9d4ea981bde21a2662f4
```

**Qwen3.5 4B Ollama manifest digest**

```text
2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd
```

**VieNeu upstream commit**

```text
36c4b501b0634a8f59805e6b529a058fbd30190b
```

**VieNeu backbone revision**

```text
2da0efab622a1722125991736524f080b751ef5b
```

**MOSS Audio Tokenizer revision**

```text
ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae
```

---

## 6. RAG và quản lý tri thức

RAG service chịu trách nhiệm cho hai flow chính.

### Knowledge flow

```text
Document
   ↓
detect / parse
   ↓
normalize
   ↓
structure-aware chunking
   ↓
dense embedding + sparse index
   ↓
verify parity
   ↓
activate document version
```

### Retrieval flow

```text
Query
  ↓
Dense retrieval
  ├──────────────┐
  │              │
  └→ Sparse retrieval
         ↓
      Fusion
         ↓
 optional rerank
         ↓
 Top-K + provenance
```

### Retrieval modes

- `dense`
- `sparse`
- `hybrid`

Reranking được bật hoặc tắt độc lập.

### Cấu hình runtime mặc định

```text
top_k       = 5
candidate_k = 50
fusion      = RRF
rrf_k       = 60
```

Benchmark RAG sử dụng cấu hình đánh giá riêng và có các metric đến Top-10.

### Chunking

Cấu hình mặc định:

```text
fallback_max_tokens = 384
overlap_tokens      = 48
reserved_tokens     = 32
min_chunk_tokens    = 8
preserve_tables     = true
```

Chunk giữ lại provenance và metadata cấu trúc khi parser cung cấp được, ví dụ:

- document ID;
- filename;
- page;
- heading;
- heading path;
- section;
- table;
- sheet;
- row;
- slide.

### Grounding policy

Với câu hỏi mang tính factual:

- có context phù hợp → context và provenance được đưa vào quá trình sinh câu trả lời;
- không có context phù hợp → hệ thống có thể sử dụng **deterministic no-answer**, thay vì ép LLM tự suy đoán.

Các tương tác xã hội như chào hỏi không bị ép qua cơ chế no-answer.

---

## 7. Thiết kế realtime

### 7.1. Control WebSocket

```text
/ws/turn/{turn_id}
```

Control stream mang các sự kiện:

- stage;
- LLM delta;
- grounding;
- source/provenance;
- done;
- error;
- stream closed.

Control path hỗ trợ cursor reconnect.

### 7.2. Audio WebSocket

```text
/ws/turn/{turn_id}/audio
```

Audio được truyền bằng binary frame chứa PCM16.

Mỗi frame có sequence riêng để:

- kiểm tra continuity;
- hỗ trợ reconnect;
- loại stale PCM.

### 7.3. AudioWorklet

Browser không tạo một lượng lớn `AudioBufferSourceNode`.

Thay vào đó:

```text
WebSocket PCM
   ↓
AudioWorklet
   ↓
bounded ring buffer
   ↓
prebuffer
   ↓
speaker
```

Khi buffer bị hụt, playback chuyển sang trạng thái rebuffer và đợi đủ audio cushion trước khi tiếp tục phát.

### 7.4. Barge-in

Khi VAD phát hiện người dùng nói trong lúc trợ lý đang phát:

1. browser flush audio đang chờ phát;
2. turn hiện tại được cancel/interrupted;
3. stale control và stale PCM bị loại;
4. audio mới được gửi tới STT;
5. một turn mới được tạo.

---

## 8. Giao diện

Web UI gồm ba khu vực chính.

### Trợ lý

- chat bằng văn bản;
- Push-to-Talk;
- VAD;
- tải file audio;
- chọn retrieval mode;
- cấu hình Top-K;
- bật/tắt rerank;
- chọn voice;
- realtime telemetry.

### Kho tri thức

- nạp tài liệu;
- tùy chọn document ID;
- xem danh sách tài liệu;
- xem trạng thái;
- xem số chunk;
- xóa tài liệu.

### Hệ thống

- service health;
- runtime profile;
- model identity;
- model revision;
- device;
- dtype;
- latency snapshot;
- TTS runtime status.

---

## 9. Runtime profiles

Runtime profile được khai báo tại:

[`config/runtime_profiles.json`](config/runtime_profiles.json)

| Profile | STT | Embedding / Reranker | LLM | TTS |
|---|---|---|---|---|
| `local-low-vram` | CUDA FP16 | CPU FP32 | CPU | CPU / ONNX |
| `local-cpu` | CPU FP32 | CPU FP32 | CPU | CPU / ONNX |
| `local-balanced` | CPU FP32 | CPU FP32 | adaptive GPU offload | CPU / ONNX |
| `colab-t4` | CUDA FP16 | CUDA FP16 | adaptive GPU offload | CPU / ONNX |
| `server-gpu` | CUDA FP16 | CUDA FP16 | adaptive GPU offload | CPU / ONNX |

Không profile nào phụ thuộc vào device `auto` cho các model chính.

Device placement được xác định trước và có thể kiểm tra lại qua health endpoint.

---

## 10. Service ports

### Windows

| Service | Port |
|---|---:|
| STT | 8100 |
| RAG | 8101 |
| LLM | 8102 |
| TTS | 8103 |
| Web gateway | 7860 |

### Colab / Linux server

| Service | Port |
|---|---:|
| STT | 8000 |
| RAG | 8001 |
| LLM | 8002 |
| TTS | 8003 |
| Web gateway | 7860 |

Ollama mặc định sử dụng:

```text
11434
```

---

## 11. Cấu trúc repository

```text
.
├── acceptance/
├── app/
│   └── voice_app/
├── config/
│   ├── model_manifest.json
│   └── runtime_profiles.json
├── docs/
│   ├── ARCHITECTURE.md
│   └── RELEASE_STATUS.md
├── scripts/
│   ├── setup_windows.ps1
│   ├── setup_colab.sh
│   ├── prefetch_tts.py
│   ├── start_services.py
│   ├── verify_services.py
│   ├── verify_product.py
│   └── benchmark_tts_v10.py
├── services/
│   ├── stt/
│   ├── rag/
│   ├── llm/
│   └── tts/
├── tests/
├── PROVENANCE.md
└── README.md
```

---

# 12. Cài đặt trên Windows

## Yêu cầu

- Windows 10 hoặc Windows 11;
- Git;
- PowerShell;
- Ollama;
- NVIDIA GPU là tùy chọn tùy profile;
- Internet ở lần setup đầu để tải dependencies và model.

## Clone repository

```powershell
git clone https://github.com/ntd2k5-2005/ai-voice-assistant-fix.git
cd ai-voice-assistant-fix
```

Để tái lập đúng source baseline hiện tại:

```powershell
git checkout 5f9626bab6532c606713d71f2ae5537dee7bd437
```

## Setup môi trường

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

Script thực hiện:

- cài Python 3.12 qua `uv` nếu cần;
- tạo virtual environment riêng cho từng service;
- cài STT dependencies;
- cài RAG;
- cài LLM service;
- clone VieNeu đúng upstream commit;
- dựng TTS environment bằng `uv sync --frozen`;
- cài app/web dependencies;
- pull `qwen3.5:4b`;
- xác minh Ollama model digest.

## Prefetch TTS artifacts

Trên clean machine, nên chạy:

```powershell
.venvs\tts\Scripts\python.exe scripts\prefetch_tts.py `
  --profile local-balanced
```

Script tải và xác minh đúng:

- VieNeu backbone;
- MOSS Audio Tokenizer;
- pinned revisions trong model manifest.

## Start full stack

```powershell
.venvs\app\Scripts\python.exe scripts\start_services.py `
  --profile local-balanced `
  --with-web
```

Sau khi các service sẵn sàng:

```text
http://127.0.0.1:7860
```

---

# 13. Chạy trên Colab / Linux

Clone:

```bash
git clone https://github.com/ntd2k5-2005/ai-voice-assistant-fix.git
cd ai-voice-assistant-fix
git checkout 5f9626bab6532c606713d71f2ae5537dee7bd437
```

Setup:

```bash
bash scripts/setup_colab.sh
```

Start:

```bash
.venvs/app/bin/python scripts/start_services.py \
  --profile colab-t4 \
  --with-web
```

`setup_colab.sh` đã bao gồm bước prefetch các TTS artifact được pin.

---

# 14. Offline / cache-only runtime

Sau khi model đã được tải đầy đủ, có thể ép Hugging Face chỉ sử dụng local cache.

Windows:

```powershell
$env:RUNTIME_OFFLINE = "true"

.venvs\app\Scripts\python.exe scripts\start_services.py `
  --profile local-balanced `
  --with-web
```

`RUNTIME_OFFLINE=true` không có nghĩa toàn bộ máy bị ngắt Internet.

Nó có nghĩa runtime không được phép âm thầm tải checkpoint Hugging Face mới trong quá trình vận hành.

---

# 15. Verification

## Kiểm tra model services

Windows:

```powershell
.venvs\app\Scripts\python.exe scripts\verify_services.py `
  --profile local-balanced
```

Colab/Linux:

```bash
.venvs/app/bin/python scripts/verify_services.py \
  --profile colab-t4
```

Verifier kiểm tra:

- STT health;
- RAG health;
- LLM health;
- TTS health;
- model manifest ID;
- runtime profile;
- device placement;
- LLM model digest;
- VieNeu backbone revision;
- audio codec revision;
- TTS realtime contract.

## Functional product verification

```powershell
.venvs\app\Scripts\python.exe scripts\verify_product.py `
  --profile local-balanced `
  --include-voice
```

Kiểm tra bao gồm:

- conversation creation;
- deterministic no-answer;
- upload knowledge;
- grounded answer;
- provenance;
- delete knowledge;
- multi-turn history;
- cancellation;
- barge-in ownership;
- stale audio leakage;
- WebSocket control;
- PCM media stream;
- synthetic voice round-trip.

Đây là **functional integration verification**, không phải benchmark latency đầu-cuối.

---

# 16. Benchmark và đánh giá

Các benchmark được tổ chức theo từng thành phần nhằm tách riêng các yếu tố chất lượng, độ trễ và chi phí xử lý.

---

## 16.1. STT

**Model**

```text
vinai/PhoWhisper-base
```

**Dataset**

```text
VIVOS — test split
```

**Thiết lập**

```text
Samples      = 760
Audio        ≈ 44,76 phút
Language     = vi
Task         = transcribe
Precision    = FP16
Device       = CUDA
```

### Kết quả

| Metric | Giá trị |
|---|---:|
| Samples | 760 / 760 |
| WER | 8,40% |
| CER | 3,80% |
| RTF | 0,054 |

RTF nhỏ hơn 1 cho thấy thời gian suy luận thấp hơn thời lượng audio đầu vào trong điều kiện benchmark.

---

## 16.2. RAG

**Dataset**

```text
YuITC/Vietnamese-Legal-Documents
```

**Subset benchmark**

```text
Documents = 2000
Queries   = 100
Seed      = 42
```

**Retrieval configuration**

```text
candidate_k = 50
```

Benchmark đánh giá:

- Recall@5;
- Recall@10;
- Hit@5;
- MRR@10;
- nDCG@10;
- mean latency;
- p95 latency.

### Latency và Hit@5

| Phương pháp | Mean latency | p95 latency | Hit@5 |
|---|---:|---:|---:|
| BM25 / Sparse | 53,75 ms | 71,91 ms | 0,86 |
| Dense | 165,86 ms | 247,65 ms | 0,91 |
| Hybrid | 198,62 ms | 300,14 ms | **0,93** |
| Hybrid + Rerank | 667,54 ms | 769,19 ms | 0,92 |

Kết quả cho thấy trade-off giữa độ trễ và chất lượng retrieval:

- BM25 có latency thấp nhất;
- Dense retrieval cải thiện khả năng khớp ngữ nghĩa;
- Hybrid đạt Hit@5 cao nhất trong phép đo;
- reranking tăng đáng kể chi phí xử lý và phù hợp hơn khi chất lượng xếp hạng được ưu tiên hơn latency.

Bộ dữ liệu pháp luật được dùng để đánh giá độc lập khả năng retrieval vì có query và relevance labels phù hợp cho các metric Information Retrieval.

Kho tri thức của ứng dụng là một tầng đánh giá khác, tập trung vào:

- document ingestion;
- retrieval;
- provenance;
- document lifecycle;
- grounded response.

---

## 16.3. LLM

**Model**

```text
qwen3.5:4b
```

**Runtime**

```text
Ollama
```

Các metric được thu thập:

- TTFT — Time To First Token;
- TBT — Time Between Tokens;
- total generation time;
- output token count;
- token throughput.

### Cấu hình benchmark

```text
temperature = 0.7
top_p       = 0.9
max_tokens  = 150
```

Một prompt trong benchmark sử dụng:

```text
max_tokens = 200
```

### Runtime defaults

```text
temperature = 0.3
top_p       = 0.9
max_tokens  = 300
```

Hiệu năng của LLM phụ thuộc đáng kể vào lượng layer được Ollama offload lên GPU.

Vì vậy runtime verification sử dụng:

```text
health snapshot
ollama ps
```

để xác nhận placement thực tế thay vì suy luận chỉ từ cấu hình.

---

## 16.4. TTS

**Model**

```text
VieNeu-TTS v3 Turbo
```

**Dataset**

```text
UIT-VSFC — test split
```

**Sample selection**

Từ tập test:

1. chọn các câu dài hơn 50 ký tự;
2. câu phải chứa dấu phẩy;
3. lấy 250 câu đầu tiên thỏa điều kiện.

Tất cả chiến lược chunking được đánh giá trên **cùng một tập 250 câu**.

**Audio output**

```text
48 kHz raw PCM
```

**Môi trường benchmark**

```text
Google Colab T4
```

Runtime của bản release sử dụng:

```text
VieNeu
ONNX backend
CPU inference
```

### Các chiến lược được đánh giá

1. đọc toàn bộ câu;
2. ngắt tại dấu phẩy;
3. ép ngắt theo khoảng 5 từ;
4. Smart Chunk với ngưỡng khoảng 40 ký tự;
5. Smart Chunk kết hợp lookahead context.

### Kết quả

| Phương pháp | TTFB mean | RTF | Ký tự/chunk | ASR accuracy | Listening score |
|---|---:|---:|---:|---:|---:|
| Đọc cả câu dài | 4041 ms | **0,691** | 103 | **91,4%** | **4,5/5** |
| Ngắt ở dấu phẩy | 1556 ms | 0,713 | 36 | 81,2% | 4,4/5 |
| Ép ngắt 5 từ | **981 ms** | 0,707 | 21 | 79,1% | 3,9/5 |
| Smart Chunk ≥ 40 ký tự | 2871 ms | 0,710 | 70 | 90,7% | **4,5/5** |
| Smart Chunk + Lookahead Context | 4039 ms | 0,734 | 70 | 89,2% | **4,5/5** |

### TTFB

**Time To First Byte** đo thời gian từ lúc bắt đầu yêu cầu tổng hợp đến khi hệ thống nhận được phần dữ liệu âm thanh đầu tiên.

TTFB thấp giúp giảm thời gian chờ trước khi người dùng bắt đầu nghe phản hồi.

### RTF

Real-Time Factor:

```text
RTF = synthesis_time / generated_audio_duration
```

Nếu:

```text
RTF < 1
```

thì tổng thời gian sinh audio thấp hơn thời lượng audio được tạo.

### ASR accuracy

Âm thanh do TTS sinh được đưa qua **OpenAI Whisper Small** để nhận dạng lại thành văn bản.

Sau đó so sánh transcript với văn bản đầu vào.

Metric được tính:

```text
ASR accuracy = 1 - WER
```

Chỉ số này phản ánh mức độ intelligibility/nội dung được phát đúng theo góc nhìn của một ASR evaluator.

Nó không đại diện đầy đủ cho:

- độ tự nhiên;
- cảm xúc;
- chất giọng;
- độ phù hợp ngữ điệu.

### Internal listening score

Khoảng **4–5 thành viên trong nhóm** nghe các mẫu âm thanh và đánh giá chất lượng cảm nhận trên thang:

```text
1 → 5
```

Điểm báo cáo là giá trị trung bình nội bộ.

Do quy mô người nghe nhỏ và chưa phải một listening study chuẩn hóa, dự án gọi chỉ số này là:

> **internal listening score (MOS-style)**

thay vì xem đây là một MOS benchmark quy mô lớn.

### Nhận xét

Kết quả cho thấy trade-off rõ giữa latency và chất lượng:

- chunk rất ngắn giúp giảm mạnh TTFB;
- tuy nhiên chunk quá ngắn làm giảm intelligibility và điểm nghe;
- đọc toàn bộ câu giữ chất lượng tốt nhưng tăng thời gian chờ trước khi audio bắt đầu;
- Smart Chunk tạo điểm cân bằng tốt hơn giữa độ dài ngữ nghĩa và khả năng streaming;
- tất cả cấu hình có aggregate RTF dưới 1 trong benchmark.

---

# 17. Trạng thái kiểm chứng

Dự án sử dụng ba mức trạng thái.

## FUNCTIONAL_PASS

Đường code/runtime đã được kiểm tra thành công với fixture hoặc service thật.

Bao gồm:

- runtime profile;
- model identity;
- startup;
- text turn;
- voice turn;
- multi-turn conversation;
- knowledge upload/delete;
- provenance;
- deterministic no-answer;
- WebSocket control;
- PCM media transport;
- cancellation;
- stale-audio protection;
- frontend initialization.

## DEGRADED

Chức năng hoạt động nhưng một số metric realtime vẫn có thể chưa ổn định.

Ví dụ:

- TTS có aggregate RTF dưới 1 nhưng vẫn có thể xuất hiện inter-chunk gap lớn;
- synthetic TTS→STT round-trip không đảm bảo transcript hoàn hảo.

## NOT_EVALUATED

Chưa có corpus, hardware hoặc protocol đủ để đưa ra kết luận.

Ví dụ:

- STT với nhiều vùng miền;
- tiếng ồn thực tế;
- code-switch;
- tên riêng;
- calibrated RAG answerability threshold;
- speaker-stop latency trên nhiều thiết bị âm thanh thật.

Chi tiết:

[`docs/RELEASE_STATUS.md`](docs/RELEASE_STATUS.md)

---

# 18. Giới hạn hiện tại

Hệ thống hiện là một **research/product prototype**.

Một số giới hạn:

- conversation history hiện lưu in-memory;
- history bị mất khi restart;
- chưa có partial ASR theo token/từ;
- TTS continuity phụ thuộc vào tốc độ sinh chunk và audio buffer;
- speaker-stop latency khi barge-in chưa được đánh giá rộng trên nhiều thiết bị;
- chất lượng STT trong noise, accent và code-switch cần corpus chuyên biệt;
- retrieval quality phụ thuộc miền dữ liệu;
- public tunnel chỉ nên dùng cho demo;
- deployment thực tế cần authentication/reverse proxy phù hợp;
- TTS dùng một inference worker để tránh concurrent model execution chưa được chứng minh an toàn.

---

# 19. Khả năng tái lập

Để một run có thể tái lập, cần giữ đồng thời:

- Git commit;
- model manifest;
- model revision;
- model digest;
- runtime profile;
- dataset;
- dataset split;
- benchmark configuration;
- hardware;
- software version;
- seed nếu có.

Source baseline của tài liệu này:

```text
5f9626bab6532c606713d71f2ae5537dee7bd437
```

Commit:

```text
fix: make clean Colab TTS runtime reproducible
```

---

# 20. Tài liệu kỹ thuật

- [`README.md`](README.md) — tổng quan dự án.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — kiến trúc realtime.
- [`docs/RELEASE_STATUS.md`](docs/RELEASE_STATUS.md) — trạng thái kiểm chứng.
- [`PROVENANCE.md`](PROVENANCE.md) — nguồn gốc các module.
- [`config/model_manifest.json`](config/model_manifest.json) — model identity.
- [`config/runtime_profiles.json`](config/runtime_profiles.json) — runtime placement.
- [`services/rag/rag/README.md`](services/rag/rag/README.md) — RAG service.
- [`services/rag/rag/BENCHMARK_REPORT.md`](services/rag/rag/BENCHMARK_REPORT.md) — benchmark evidence của RAG service.

---

# 21. Third-party models và nguồn tham khảo

### PhoWhisper

https://huggingface.co/vinai/PhoWhisper-base

### multilingual-e5-small

https://huggingface.co/intfloat/multilingual-e5-small

### GTE multilingual reranker

https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base

### Ollama

https://ollama.com/

### VieNeu-TTS

https://github.com/pnnbao97/VieNeu-TTS

### VieNeu-TTS v3 Turbo

https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo

### MOSS Audio Tokenizer

https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX

Revision và model identity cụ thể được lưu tại:

[`config/model_manifest.json`](config/model_manifest.json)

Nguồn gốc các module:

[`PROVENANCE.md`](PROVENANCE.md)

---

# 22. Đóng góp của hệ thống

Dự án tập trung vào ba đóng góp chính.

### 1. Kiến trúc AI đàm thoại cục bộ có khả năng tái lập

STT, RAG, LLM và TTS được tổ chức thành các service độc lập, với model manifest và runtime profile giúp kiểm soát model identity, placement và cấu hình triển khai.

### 2. Cơ chế điều phối hội thoại theo luồng

Control path, speech scheduling path và media path được tách riêng.

Turn ownership, generation ownership và playback epoch giúp duy trì tính nhất quán khi:

- streaming;
- cancellation;
- VAD;
- barge-in;
- reconnect.

### 3. Hỏi đáp tri thức có kiểm soát

Hệ thống kết hợp:

- dense retrieval;
- sparse retrieval;
- hybrid fusion;
- reranking;
- provenance;
- document lifecycle;
- deterministic no-answer.

Điều này giúp giảm phụ thuộc vào khả năng suy đoán tự do của LLM khi truy vấn yêu cầu thông tin từ kho tri thức.

---

# 23. Phạm vi nghiên cứu

Dự án tập trung vào bài toán:

> **Thiết kế và đánh giá một hệ thống AI đàm thoại tiếng Việt có thể vận hành theo cơ chế streaming và triển khai trong môi trường cục bộ, đồng thời hỗ trợ truy xuất tri thức có nguồn tham chiếu.**

Điểm mới của công trình nằm ở cách tổ chức và điều phối các thành phần AI thành một pipeline hội thoại thống nhất, thay vì chỉ đánh giá từng mô hình riêng lẻ.

Các mục tiêu ưu tiên:

- khả năng vận hành;
- khả năng tái lập;
- kiểm soát dữ liệu;
- realtime streaming;
- grounding;
- khả năng thay thế model theo điều kiện phần cứng.

---

# 24. Ghi chú triển khai

Khi sử dụng trong môi trường nội bộ có dữ liệu nhạy cảm:

- ưu tiên chạy trong LAN hoặc private infrastructure;
- không mở public tunnel nếu không cần;
- kiểm soát quyền truy cập kho tri thức;
- pin model revision;
- giữ model manifest;
- bật cache-only runtime sau khi setup hoàn tất;
- kiểm tra service health trước khi nghiệm thu;
- không dùng benchmark từ môi trường/hardware khác như thể chúng hoàn toàn tương đương.

---

**Research prototype for Vietnamese realtime conversational AI and local knowledge assistance.**
