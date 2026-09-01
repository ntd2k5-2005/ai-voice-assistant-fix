from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]


def test_root_wheel_contains_and_imports_rag_from_outside_source_tree(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(tmp_path.glob("robust_rag_service-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "rag/__init__.py" in names
    assert "rag/config.py" in names
    assert "rag/service.py" in names
    assert "rag/api/app.py" in names
    assert "rag/api/schemas.py" in names
    assert not any(name.startswith("ragflow-main/") for name in names)

    script = (
        "import pathlib,sys;"
        f"sys.path.insert(0,{str(wheel)!r});"
        "import rag;"
        "location=pathlib.PurePath(rag.__file__).as_posix();"
        "print(location);"
        f"assert {wheel.name!r} in location"
    )
    imported = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
