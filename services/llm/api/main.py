from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from llm import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    GenerationTimeoutError,
    LLMService,
    ModelNotFoundError,
    OllamaUnavailableError,
    Settings,
)
from llm.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
settings = Settings()
llm_service: LLMService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_service
    logger.info(
        "Starting LLM API model=%s ollama=%s",
        settings.model_name,
        settings.ollama_url,
    )
    llm_service = LLMService(settings)
    health = await run_in_threadpool(llm_service.health_check)
    if health["ollama_available"] and health["model_loaded"]:
        logger.info("Model %s ready", settings.model_name)
    else:
        logger.warning("LLM startup health: %s", health)
    yield
    logger.info("Shutting down LLM API")


app = FastAPI(
    title="LLM API Service",
    description="Qwen/Ollama API with blocking and NDJSON streaming chat.",
    version="1.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm_service() -> LLMService:
    if llm_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return llm_service


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started = datetime.now()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    if settings.log_request_id:
        elapsed_ms = (datetime.now() - started).total_seconds() * 1000
        logger.info(
            "[%s] %s %s completed in %.2fms status=%s",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
            response.status_code,
        )
    return response


@app.get("/health/live", tags=["Health"])
async def liveness():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/health/ready", response_model=HealthResponse, tags=["Health"])
async def readiness(service: LLMService = Depends(get_llm_service)):
    health = await run_in_threadpool(service.health_check)
    if not health["ollama_available"]:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unavailable: {health.get('error')}",
        )
    if not health["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model {service.model_name} not loaded",
        )
    return HealthResponse(
        status="healthy",
        model=service.model_name,
        ollama_available=True,
        model_loaded=True,
        model_digest=health.get("model_digest"),
        quantization_level=health.get("quantization_level"),
        parameter_size=health.get("parameter_size"),
        configured_placement=health.get("configured_placement"),
        configured_num_gpu=health.get("configured_num_gpu"),
        active_processor=health.get("active_processor"),
        active_size_bytes=health.get("active_size_bytes"),
        active_size_vram_bytes=health.get("active_size_vram_bytes"),
        runtime_profile=health.get("runtime_profile"),
        model_manifest_id=health.get("model_manifest_id"),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/v1/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(
    request: GenerateRequest,
    req: Request,
    service: LLMService = Depends(get_llm_service),
):
    request_id = req.state.request_id
    try:
        result = await run_in_threadpool(
            service.generate,
            request.prompt,
            request.temperature,
            request.top_p,
            request.max_tokens,
            settings.enable_metrics,
            request_id,
        )
        return GenerateResponse(
            response=result["response"],
            model=service.model_name,
            metrics=result.get("metrics"),
            timestamp=datetime.now().isoformat(),
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GenerationTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] generation failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {exc}",
        ) from exc


@app.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_conversation(
    request: ChatRequest,
    req: Request,
    service: LLMService = Depends(get_llm_service),
):
    request_id = req.state.request_id
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    try:
        result = await run_in_threadpool(
            service.chat,
            messages,
            request.temperature,
            request.top_p,
            request.max_tokens,
            request_id,
        )
        from llm.schemas import Message

        return ChatResponse(
            message=Message(**result["message"]),
            model=service.model_name,
            timestamp=datetime.now().isoformat(),
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GenerationTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] chat failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        ) from exc


@app.post("/v1/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    req: Request,
    service: LLMService = Depends(get_llm_service),
):
    """Stream assistant deltas as newline-delimited JSON."""
    request_id = req.state.request_id
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]

    def source():
        try:
            for event in service.chat_stream(
                messages,
                request.temperature,
                request.top_p,
                request.max_tokens,
                request_id,
            ):
                event = {
                    **event,
                    "model": service.model_name,
                    "request_id": request_id,
                }
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("[%s] streaming chat failed", request_id)
            yield json.dumps(
                {
                    "type": "error",
                    "message": str(exc),
                    "request_id": request_id,
                    "model": service.model_name,
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        iterate_in_threadpool(source()),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "LLM API Service",
        "version": "1.2.0",
        "model": settings.model_name,
        "endpoints": {
            "docs": "/docs",
            "health_live": "/health/live",
            "health_ready": "/health/ready",
            "generate": "/v1/generate",
            "chat": "/v1/chat",
            "chat_stream": "/v1/chat/stream",
        },
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
