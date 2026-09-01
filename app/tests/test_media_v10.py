import time
import re
from pathlib import Path
import numpy as np
from voice_app.config import AppSettings
from voice_app.contracts import RAGContext,RAGLatency,RAGResult,ServiceStatus,STTResult
from voice_app.media_v10 import V10MediaPipeline
from voice_app.orchestrator import VoiceOrchestrator
from voice_app.web import AudioPacket, PCM_WS_HEADER, TurnManager, encode_pcm_websocket_packet

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
    def stream_native_pcm(self,text,*,voice_index,trace_id=None,unit_index=None):
        self.calls.append(text); started=time.perf_counter()
        time.sleep(.03)
        for _ in range(2):
            raw=(np.ones(100,dtype="<f4")*.1).tobytes(); yield {"type":"audio","raw":raw,"duration_ms":100,"ttfb_ms":30,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}
        total=(time.perf_counter()-started)*1000; yield {"type":"done","rtf":total/200,"ttfb_ms":30,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}
    def save_pcm_wav(self,raw,*,prefix): self.out.mkdir(parents=True,exist_ok=True); p=self.out/f"{prefix}.wav"; p.write_bytes(b"RIFF"+raw); return str(p)
    def close(self): pass


class BelowRealtimeTTS(SlowTTS):
    def stream_native_pcm(self,text,*,voice_index,trace_id=None,unit_index=None):
        self.calls.append(text)
        started=time.perf_counter()
        for _ in range(2):
            time.sleep(.02)
            raw=(np.ones(10,dtype="<f4")*.1).tobytes()
            yield {"type":"audio","raw":raw,"frame_count":10,"duration_ms":10,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}
        total=(time.perf_counter()-started)*1000
        yield {"type":"done","rtf":total/20,"ttfb_ms":20,"streaming_mode":"frame-pcm","upstream_backend":"onnx","profile":"realtime"}

def build(tmp):
    s=AppSettings(_env_file=None,tts_sample_rate=1000,tts_first_unit_min_chars=20,tts_first_unit_target_chars=45,tts_first_unit_max_chars=70,tts_steady_unit_min_chars=30,tts_steady_unit_target_chars=70,tts_steady_unit_max_chars=100,tts_sentence_queue_max=1,tts_output_dir=tmp)
    return VoiceOrchestrator(s,stt_client=FakeSTT(),rag_client=FakeRAG(),llm_client=FastLLM(),tts_client=SlowTTS(tmp))

def test_v10_decouples_llm_reader_from_tts_backpressure(tmp_path):
    orchestrator = build(tmp_path)
    events=list(V10MediaPipeline(orchestrator).stream_turn(query_text="V10?",rag_mode="dense",rerank=False,top_k=1))
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
    pcm = [event for event in events if event["type"] == "pcm"]
    assert [event["chunk_index"] for event in pcm] == list(range(1, len(pcm) + 1))
    assert all(event["frame_count"] == 100 for event in pcm)
    expected = "".join(
        event["content"]
        for event in FastLLM().chat_stream([])
        if event["type"] == "delta"
    )
    assert re.sub(r"\s+", " ", " ".join(orchestrator.tts.calls)).strip() == re.sub(r"\s+", " ", expected).strip()

def test_pcm_conversion():
    raw=np.array([-1.,0.,1.],dtype="<f4").tobytes(); vals=np.frombuffer(TurnManager._pcm_float32_to_int16(raw),dtype="<i2")
    assert vals.tolist()==[-32767,0,32767]


def test_pcm_websocket_frame_preserves_sequence_and_samples():
    raw = np.array([-7, 0, 9], dtype="<i2").tobytes()
    packet = AudioPacket(raw=raw, meta={
        "chunk_index": 17,
        "unit_index": 4,
        "unit_chunk_index": 2,
        "frame_count": 3,
        "t0_model_yield_ns": 10,
        "t1_service_yield_ns": 20,
        "t2_client_receive_ns": 30,
        "t3_media_enqueue_ns": 40,
        "t4_web_audio_enqueue_ns": 50,
    })
    encoded = encode_pcm_websocket_packet(packet, sent_ns=60)
    fields = PCM_WS_HEADER.unpack_from(encoded)
    assert fields[:7] == (b"V10P", 1, PCM_WS_HEADER.size, 17, 4, 2, 3)
    assert fields[7:] == (10, 20, 30, 40, 50, 60)
    assert encoded[PCM_WS_HEADER.size:] == raw


def test_turn_manager_keeps_pcm_order_through_web_audio_queue(tmp_path):
    manager = TurnManager(build(tmp_path))
    session = manager.start(query_text="V10?", rag_mode="dense", rerank=False, top_k=1)
    packets = []
    while True:
        packet = session.audio_queue.get(timeout=3)
        if packet is None:
            break
        packets.append(packet)
    assert session.finished.wait(timeout=1)
    assert [packet.meta["chunk_index"] for packet in packets] == list(range(1, len(packets) + 1))
    assert all(packet.meta["t4_web_audio_enqueue_ns"] > 0 for packet in packets)
    assert b"".join(packet.raw for packet in packets)


def test_pipeline_explicitly_flags_below_realtime_tts(tmp_path):
    orchestrator = build(tmp_path)
    orchestrator.tts = BelowRealtimeTTS(tmp_path)
    events = list(
        V10MediaPipeline(orchestrator).stream_turn(
            query_text="V10?", rag_mode="dense", rerank=False, top_k=1
        )
    )
    meta = next(event["meta"] for event in events if event["type"] == "tts_done")
    assert meta["aggregate_rtf"] > 1.0
    assert meta["producer_throughput_status"] == "insufficient"

from voice_app.web import _safe_filename

def test_safe_filename_preserves_original_basename():
    assert _safe_filename('../../HUIT_Knowledge_2026.docx', 'document.bin') == 'HUIT_Knowledge_2026.docx'
