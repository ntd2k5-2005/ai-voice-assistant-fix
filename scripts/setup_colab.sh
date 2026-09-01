#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="$ROOT/.venvs"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
OLLAMA_VERSION="${OLLAMA_VERSION:-0.32.15}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:4b}"
RUNTIME_JSON="$ROOT/services/tts/runtime.json"

VIENEU_REPO="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository"])' "$RUNTIME_JSON")"
VIENEU_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$RUNTIME_JSON")"
VIENEU_DIR="$ROOT/.runtime/VieNeu-TTS"

echo "==> System dependencies"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd curl ca-certificates pciutils ffmpeg libsndfile1 git

python3 -m pip install -U uv
UV="$(command -v uv)"
"$UV" python install "$PYTHON_VERSION"

mkdir -p "$VENVS" "$ROOT/.runtime"
for name in stt rag llm app; do
  "$UV" venv --python "$PYTHON_VERSION" --seed "$VENVS/$name"
done

echo "==> STT"
"$UV" pip install --python "$VENVS/stt/bin/python" -r "$ROOT/services/stt/requirements.txt" -r "$ROOT/services/stt/requirements-test.txt"

echo "==> RAG"
"$UV" pip install --python "$VENVS/rag/bin/python" -e "$ROOT/services/rag[api]"

echo "==> LLM"
"$UV" pip install --python "$VENVS/llm/bin/python" -e "$ROOT/services/llm[api,test]"

echo "==> VieNeu V10 locked realtime runtime"
if [[ ! -d "$VIENEU_DIR/.git" ]]; then
  rm -rf "$VIENEU_DIR"
  git clone "$VIENEU_REPO" "$VIENEU_DIR"
fi
git -C "$VIENEU_DIR" fetch --depth 1 origin "$VIENEU_COMMIT"
git -C "$VIENEU_DIR" checkout --detach "$VIENEU_COMMIT"
ACTUAL_COMMIT="$(git -C "$VIENEU_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$VIENEU_COMMIT" ]]; then
  echo "VieNeu commit mismatch: expected=$VIENEU_COMMIT actual=$ACTUAL_COMMIT" >&2
  exit 1
fi
rm -rf "$VENVS/tts"
UV_PROJECT_ENVIRONMENT="$VENVS/tts" UV_PYTHON="$PYTHON_VERSION" "$UV" sync --project "$VIENEU_DIR" --frozen
# Service-only packages are installed after the locked VieNeu environment. They
# do not replace VieNeu/onnxruntime and keep the upstream CPU lock intact.
"$UV" pip install --python "$VENVS/tts/bin/python" "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1" "pydantic>=2.8,<3" "pytest>=8,<10"

echo "==> App V10"
"$UV" pip install --python "$VENVS/app/bin/python" -e "$ROOT/app[demo,dev]"

echo "==> Ollama $OLLAMA_VERSION"
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_VERSION" sh
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  nohup ollama serve > "$ROOT/ollama.log" 2>&1 &
  sleep 3
fi
ollama pull "$OLLAMA_MODEL"

echo
echo "Setup complete. VieNeu pinned at $VIENEU_COMMIT with uv sync --frozen."
echo "Start services: $VENVS/app/bin/python $ROOT/scripts/start_services.py"
echo "Start V10 UI:    cd $ROOT/app && ../.venvs/app/bin/python -m voice_app.web"
