import time
from pathlib import Path
import numpy as np
from voice_app.config import AppSettings
from voice_app.contracts import RAGContext,RAGLatency,RAGResult,ServiceStatus,STTResult
from voice_app.media_v9 import V9MediaPipeline
from voice_app.orchestrator import VoiceOrchestrator
from voice_app.web import TurnManager

class FakeSTT:
    def health(self): return ServiceStatus(service="STT",reachable=True,ready=True)
    def transcribe(self,path): return STTResult(text="Kiểm thử V9",client_latency_ms=2)
    def close(self): pass
class FakeRAG:
    def health(self): return ServiceStatus(service="RAG",reachable=True,ready=True)
    def retrieve(self,query,**kwargs): return RAGResult(query=query,mode=kwargs["mode"],rerank_requested=False,rerank_applied=False,contexts=[RAGContext(rank=1,chunk_id="c1",document_id="d1",text="EUREKA V9",score=.9,score_source="dense")],latency=RAGLatency(total_ms=1),client_latency_ms=2)
    def close(self): pass
class SlowLLM:
    def health(self): return ServiceStatus(service="LLM",reachable=True,ready=True)
    def chat_stream(self,messages,**kwargs):
        yield {"type":"delta","content":"Đây là một mệnh đề đủ dài để phát sớm, và phần còn lại vẫn đang được LLM tạo ra","client_ttft_ms":1}
        time.sleep(.08)
        yield {"type":"delta","content":" trước khi kết thúc câu hoàn chỉnh.","client_ttft_ms":1}
        yield {"type":"done","metrics":{"ttft_ms":1,"total_latency_ms":90},"client_total_ms":90}
    def close(self): pass
class FakeTTS:
    def __init__(self,out): self.out=Path(out); self.calls=[]
    def health(self): return ServiceStatus(service="TTS",reachable=True,ready=True,details={"pipeline":"v9","profile":"realtime","backend":"onnx","streaming_mode":"frame-pcm"})
    def stream_native_pcm(self,text,*,voice_index):
        self.calls.append(text); started=time.perf_counter()
        for _ in range(2):
            time.sleep(.004); raw=(np.ones(100,dtype="<f4")*.1).tobytes(); yield {"type":"audio","raw":raw,"duration_ms":100,"ttfb_ms":4,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}
        total=(time.perf_counter()-started)*1000; yield {"type":"done","rtf":total/200,"ttfb_ms":4,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}
    def save_pcm_wav(self,raw,*,prefix): self.out.mkdir(parents=True,exist_ok=True); p=self.out/f"{prefix}.wav"; p.write_bytes(b"RIFF"+raw); return str(p)
    def close(self): pass

def build(tmp):
    s=AppSettings(_env_file=None,tts_sample_rate=1000,tts_speech_unit_min_chars=20,tts_speech_unit_target_chars=45,tts_speech_unit_max_chars=80,tts_output_dir=tmp)
    return VoiceOrchestrator(s,stt_client=FakeSTT(),rag_client=FakeRAG(),llm_client=SlowLLM(),tts_client=FakeTTS(tmp))

def test_v9_emits_progress_and_pcm_before_llm_done(tmp_path):
    events=list(V9MediaPipeline(build(tmp_path)).stream_turn(query_text="V9?",rag_mode="dense",rerank=False,top_k=1))
    kinds=[e["type"] for e in events]
    assert "stage" in kinds and "prepared" in kinds and "speech_unit_queued" in kinds
    assert kinds.index("pcm") < kinds.index("llm_done")
    meta=next(e for e in events if e["type"]=="tts_done")["meta"]
    assert meta["pipeline_version"]=="v9"
    assert meta["playout_owner"]=="browser-audioworklet-ring-buffer"
    assert meta["server_playout_buffering"] is False
    assert meta["max_speech_unit_chars"] <= 80

def test_pcm_conversion():
    raw=np.array([-1.,0.,1.],dtype="<f4").tobytes(); vals=np.frombuffer(TurnManager._pcm_float32_to_int16(raw),dtype="<i2")
    assert vals.tolist()==[-32767,0,32767]

from voice_app.web import _safe_filename

def test_safe_filename_preserves_original_basename():
    assert _safe_filename('../../HUIT_Knowledge_2026.docx', 'document.bin') == 'HUIT_Knowledge_2026.docx'
