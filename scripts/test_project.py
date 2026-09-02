from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENVS = ROOT / ".venvs"


def vpython(name: str) -> Path:
    path = VENVS / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing venv for {name}: {path}")


def run_pytest(python: Path, cwd: Path, *targets: str) -> None:
    env=os.environ.copy(); env["PYTHONPATH"]=str(cwd)
    print(f"\n==> {cwd.relative_to(ROOT) or '.'}: {' '.join(targets)}")
    subprocess.run([str(python), "-m", "pytest", *targets], cwd=cwd, env=env, check=True)


def main() -> None:
    run_pytest(vpython("app"), ROOT, "tests")
    run_pytest(vpython("app"), ROOT/"app", "tests")
    run_pytest(vpython("stt"), ROOT/"services/stt", "tests")
    run_pytest(vpython("rag"), ROOT/"services/rag", "rag/tests")
    run_pytest(vpython("llm"), ROOT/"services/llm", "tests")
    run_pytest(vpython("tts"), ROOT/"services/tts", "tests")
    print("\nPASS: module-isolated project test suites")

if __name__ == "__main__": main()
