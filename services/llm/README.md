# LLM service

Stateless Qwen/Ollama API used by the Eureka voice assistant.

## Important source fixes

- Packaging is flat and valid: `services/llm/{llm,api,tests}`.
- Uses `ollama.Client(host=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)` for every call.
- `think=False` is sent to Qwen so voice chat returns final content instead of
  an empty `message.content`.
- Empty assistant content is rejected explicitly.
- Synchronous Ollama calls are run in FastAPI's thread pool.
- Generation token metrics use Ollama `eval_count/eval_duration` when available.

The HUIT/RAG system prompt is intentionally owned by `app/`, not this service.

## Install

```bash
pip install -e ".[api,test]"
```

Ollama:

```bash
ollama serve
ollama pull qwen3.5:4b
```

## Run

```bash
API_PORT=8002 python -m api.main
```

Default monorepo port: **8002**.

## Endpoints

- `GET /health/live`
- `GET /health/ready`
- `POST /v1/generate`
- `POST /v1/chat`

`/v1/chat` remains stateless: the caller sends the complete `messages[]`
required for the current turn.
