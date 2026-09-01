from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENVS = ROOT / ".venvs"
LOGS = ROOT / ".logs"


def vpython(name: str) -> Path:
    path = VENVS / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} venv Python: {path}. Run setup first.")
    return path


def wait_http(url: str, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if 200 <= r.status < 300:
                    print(f"READY {url}")
                    return
                last = f"HTTP {r.status}"
        except Exception as exc:
            last = repr(exc)
        time.sleep(2)
    raise RuntimeError(f"Timeout waiting for {url}; last={last}")


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    env = os.environ.copy()
    runtime = json.loads((ROOT / "services/tts/runtime.json").read_text(encoding="utf-8"))
    tts_env = {
        "TTS_PROFILE": "realtime",
        "TTS_WARMUP": env.get("TTS_WARMUP", "true"),
        "VIENEU_UPSTREAM_COMMIT": runtime["commit"],
        "VIENEU_UPSTREAM_REPOSITORY": runtime["repository"],
    }
    specs = [
        ("stt", [str(vpython("stt")), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"], ROOT/"services/stt", {}, "http://127.0.0.1:8000/health/ready"),
        ("rag", [str(vpython("rag")), "-m", "rag.api"], ROOT/"services/rag", {"RAG_API_HOST":"127.0.0.1","RAG_API_PORT":"8001"}, "http://127.0.0.1:8001/health/ready"),
        ("llm", [str(vpython("llm")), "-m", "api.main"], ROOT/"services/llm", {"API_HOST":"127.0.0.1","API_PORT":"8002","OLLAMA_URL":env.get("OLLAMA_URL","http://127.0.0.1:11434"),"MODEL_NAME":env.get("OLLAMA_MODEL","qwen3.5:4b")}, "http://127.0.0.1:8002/health/ready"),
        ("tts", [str(vpython("tts")), "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8003"], ROOT/"services/tts", tts_env, "http://127.0.0.1:8003/health/ready"),
    ]
    processes=[]; handles=[]
    try:
        for name,cmd,cwd,extra,health in specs:
            fh=(LOGS/f"{name}.log").open("w",encoding="utf-8"); handles.append(fh)
            print(f"START {name}: {' '.join(cmd)}")
            p=subprocess.Popen(cmd,cwd=cwd,env=env|extra,stdout=fh,stderr=subprocess.STDOUT); processes.append(p)
            wait_http(health)
        print("\nAll four model services are ready. Run V10 UI from app/ with voice_app.web.")
        while True:
            for p in processes:
                if p.poll() is not None: raise RuntimeError(f"A service exited unexpectedly with rc={p.returncode}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for p in reversed(processes):
            if p.poll() is None: p.terminate()
        for p in reversed(processes):
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill()
        for fh in handles: fh.close()

if __name__ == "__main__": main()
