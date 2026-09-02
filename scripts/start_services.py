from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from runtime_config import load_manifest, load_profile, service_environments

ROOT = Path(__file__).resolve().parents[1]
VENVS = ROOT / ".venvs"
LOGS = ROOT / ".logs"


def vpython(name: str) -> Path:
    path = VENVS / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} venv Python: {path}. Run setup first.")
    return path


def wait_http(
    url: str,
    *,
    process: subprocess.Popen | None = None,
    service: str = "service",
    timeout_s: int = 900,
) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"{service} exited during startup with rc={process.returncode}"
            )
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


def ensure_ports_available(host: str, ports: dict[str, int]) -> None:
    conflicts: list[str] = []
    for name, port in ports.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex((host, int(port))) == 0:
                conflicts.append(f"{name}={port}")
    if conflicts:
        raise RuntimeError(
            "Runtime profile ports are already in use: " + ", ".join(conflicts)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the manifest-driven V10 stack")
    parser.add_argument(
        "--profile",
        default=os.getenv("RUNTIME_PROFILE"),
        help="Runtime profile from config/runtime_profiles.json",
    )
    parser.add_argument(
        "--with-web",
        action="store_true",
        help="Also start the browser gateway using the same profile URLs",
    )
    args = parser.parse_args()
    if not args.profile:
        parser.error("--profile is required (or set RUNTIME_PROFILE explicitly)")

    LOGS.mkdir(exist_ok=True)
    env = os.environ.copy()
    manifest = load_manifest(ROOT)
    profile_name, profile = load_profile(ROOT, args.profile)
    environments = service_environments(ROOT, manifest, profile_name, profile, env)
    ports = profile["ports"]
    checked_ports = {name: int(ports[name]) for name in ("stt", "rag", "llm", "tts")}
    if args.with_web:
        checked_ports["web"] = int(ports["web"])
    ensure_ports_available("127.0.0.1", checked_ports)

    environments["tts"]["TTS_WARMUP"] = env.get("TTS_WARMUP", "true")
    specs = [
        ("stt", [str(vpython("stt")), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(ports["stt"])], ROOT/"services/stt", environments["stt"], f"http://127.0.0.1:{ports['stt']}/health/ready"),
        ("rag", [str(vpython("rag")), "-m", "rag.api"], ROOT/"services/rag", environments["rag"], f"http://127.0.0.1:{ports['rag']}/health/ready"),
        ("llm", [str(vpython("llm")), "-m", "api.main"], ROOT/"services/llm", environments["llm"], f"http://127.0.0.1:{ports['llm']}/health/ready"),
        ("tts", [str(vpython("tts")), "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(ports["tts"])], ROOT/"services/tts", environments["tts"], f"http://127.0.0.1:{ports['tts']}/health/ready"),
    ]
    if args.with_web:
        specs.append(
            ("web", [str(vpython("app")), "-m", "voice_app.web"], ROOT/"app", environments["web"], f"http://127.0.0.1:{ports['web']}/health/live")
        )
    processes=[]; handles=[]
    try:
        print(f"PROFILE {profile_name}: {profile['platform']}")
        print(f"MANIFEST {manifest['manifest_id']}")
        for name,cmd,cwd,extra,health in specs:
            fh=(LOGS/f"{name}.log").open("w",encoding="utf-8"); handles.append(fh)
            print(f"START {name}: {' '.join(cmd)}")
            p=subprocess.Popen(cmd,cwd=cwd,env=extra,stdout=fh,stderr=subprocess.STDOUT); processes.append(p)
            wait_http(health, process=p, service=name)
        if args.with_web:
            print(f"\nV10 stack ready at http://127.0.0.1:{ports['web']}")
        else:
            print("\nAll four model services are ready. Re-run with --with-web for the browser gateway.")
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
