from __future__ import annotations
import argparse, json, statistics, time
import httpx, numpy as np

def run_once(url,text,voice):
    started=time.perf_counter(); first=None; last=None; gaps=[]; parts=[]; headers={}
    with httpx.Client(timeout=httpx.Timeout(420.0)) as client:
        with client.stream("POST",f"{url.rstrip('/')}/api/tts",json={"text":text,"voice_index":voice,"pre_chunked":True,"generation_mode":"stream"}) as response:
            response.raise_for_status(); headers=dict(response.headers)
            for part in response.iter_raw():
                if not part: continue
                now=time.perf_counter(); first=first or now
                if last is not None: gaps.append((now-last)*1000.0)
                last=now; parts.append(part)
    finished=time.perf_counter(); raw=b"".join(parts)
    if not raw or len(raw)%4: raise RuntimeError(f"invalid PCM bytes: {len(raw)}")
    samples=np.frombuffer(raw,dtype="<f4"); audio_ms=samples.size/48000*1000; total_ms=(finished-started)*1000
    return {"backend":headers.get("x-tts-backend"),"upstream_commit":headers.get("x-vieneu-upstream-commit"),"streaming_mode":headers.get("x-audio-streaming"),"first_body_ms":((first-started)*1000 if first else None),"total_ms":total_ms,"audio_duration_ms":audio_ms,"rtf":total_ms/audio_ms if audio_ms else None,"network_chunks":len(parts),"max_inter_chunk_gap_ms":max(gaps) if gaps else None,"p95_inter_chunk_gap_ms":statistics.quantiles(gaps,n=20,method="inclusive")[18] if len(gaps)>1 else (gaps[0] if gaps else None)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--url",default="http://127.0.0.1:8003");p.add_argument("--voice",type=int,default=2);p.add_argument("--runs",type=int,default=3);p.add_argument("--text",default="Đây là một đoạn nói ngắn để đo tốc độ phát âm thanh thời gian thực của hệ thống.");a=p.parse_args()
    rows=[run_once(a.url,a.text,a.voice) for _ in range(a.runs)]
    for i,row in enumerate(rows,1): print(f"run {i}: "+json.dumps(row,ensure_ascii=False))
    print(json.dumps({"median_ttfb_ms":statistics.median(x["first_body_ms"] for x in rows),"median_rtf":statistics.median(x["rtf"] for x in rows),"max_gap_ms":max((x["max_inter_chunk_gap_ms"] or 0) for x in rows),"realtime_headroom":statistics.median(x["rtf"] for x in rows)<1},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
