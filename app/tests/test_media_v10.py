import time
from pathlib import Path
import numpy as np
from voice_app.config import AppSettings
from voice_app.contracts import RAGContext,RAGLatency,RAGResult,ServiceStatus,STTResult
from voice_app.media_v10 import V10MediaPipeline
from voice_app.orchestrator import VoiceOrchestrator
from voice_app.web import TurnManager

class FakeSTT:
    def health(self): return ServiceStatus(service="STT",reachable=True,ready=True)
    def transcribe(self,path): return STTResult(text="Kiểm thử V10",client_latency_ms=2)
    def close(self): pass
class FakeRAG:
    def health(self): return ServiceStatus(service="RAG",reachable=True,ready=True)
    def retrieve(self,query,**kwargs): return RAGResult(query=query,mode=kwargs["mode"],rerank_requested=False,rerank_applied=False,contexts=[RAGContext(rank=1,chunk_id="c1",document_id="d1",text="EUREKA V10",score=.9,score_source="dense")],latency=RAGLatency(total_ms=1),client_latency_ms=2)
    def close(self): pass
class FastLLM:
    def health(self): return ServiceStatus(service="LLM",reachable=True,ready=True)
    def chat_stream(self,messages,**kwargs):
        for part in [
            "Đây là phần đầu đủ dài để tạo một speech unit sớm, và LLM vẫn tiếp tục sinh nội dung. ",
            "Phần thứ hai tiếp tục được stream ngay cả khi TTS phía sau đang xử lý rất chậm. ",
            "Phần cuối hoàn tất câu trả lời.",
        ]:
            yield {"type":"delta","content":part,"client_ttft_ms":1}
        yield {"type":"done","metrics":{"ttft_ms":1,"total_latency_ms":5},"client_total_ms":5}
    def close(self): pass
class SlowTTS:
    def __init__(self,out): self.out=Path(out); self.calls=[]
    def health(self): return ServiceStatus(service="TTS",reachable=True,ready=True,details={"pipeline":"v10","profile":"realtime","backend":"onnx","streaming_mode":"frame-pcm"})
    def stream_native_pcm(self,text,*,voice_index):
        self.calls.append(text); started=time.perf_counter()
        time.sleep(.03)
        for _ in range(2):
            raw=(np.ones(100,dtype="<f4")*.1).tobytes(); yield {"type":"audio","raw":raw,"duration_ms":100,"ttfb_ms":30,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}
        total=(time.perf_counter()-started)*1000; yield {"type":"done","rtf":total/200,"ttfb_ms":30,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}
    def save_pcm_wav(self,raw,*,prefix): self.out.mkdir(parents=True,exist_ok=True); p=self.out/f"{prefix}.wav"; p.write_bytes(b"RIFF"+raw); return str(p)
    def close(self): pass

def build(tmp):
    s=AppSettings(_env_file=None,tts_sample_rate=1000,tts_first_unit_min_chars=20,tts_first_unit_target_chars=45,tts_first_unit_max_chars=70,tts_steady_unit_min_chars=30,tts_steady_unit_target_chars=70,tts_steady_unit_max_chars=100,tts_sentence_queue_max=1,tts_output_dir=tmp)
    return VoiceOrchestrator(s,stt_client=FakeSTT(),rag_client=FakeRAG(),llm_client=FastLLM(),tts_client=SlowTTS(tmp))

def test_v10_decouples_llm_reader_from_tts_backpressure(tmp_path):
    events=list(V10MediaPipeline(build(tmp_path)).stream_turn(query_text="V10?",rag_mode="dense",rerank=False,top_k=1))
    kinds=[e["type"] for e in events]
    assert "llm_delta" in kinds and "speech_unit_queued" in kinds and "pcm" in kinds
    meta=next(e for e in events if e["type"]=="tts_done")["meta"]
    assert meta["pipeline_version"]=="v10"
    assert meta["control_transport"]=="websocket"
    assert meta["llm_reader_blocked_by_tts"] is False
    assert meta["llm_reader_tts_backpressure_ms"] == 0
    assert meta["tts_queue_backpressure_ms"] >= 0
    llm_done=next(e for e in events if e["type"]=="llm_done")
    # Fast LLM can finish before a deliberately slow TTS drains its backlog.
    assert llm_done["meta"]["pipeline_done_ms"] < meta["units"][-1]["synthesis_finished_ms"]

def test_pcm_conversion():
    raw=np.array([-1.,0.,1.],dtype="<f4").tobytes(); vals=np.frombuffer(TurnManager._pcm_float32_to_int16(raw),dtype="<i2")
    assert vals.tolist()==[-32767,0,32767]

from voice_app.web import _safe_filename

def test_safe_filename_preserves_original_basename():
    assert _safe_filename('../../HUIT_Knowledge_2026.docx', 'document.bin') == 'HUIT_Knowledge_2026.docx'
