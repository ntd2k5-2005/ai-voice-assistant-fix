from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import AppSettings
from .orchestrator import VoiceOrchestrator


class TextChatRequest(BaseModel):
    query: str = Field(min_length=1)
    rag_mode: Literal["dense", "sparse", "hybrid"] | None = None
    rerank: bool | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    synthesize: bool = True
    voice_index: int | None = None


def create_app():
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import StreamingResponse
        from starlette.concurrency import iterate_in_threadpool
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI extra missing. Run: pip install -e '.[api]'"
        ) from exc

    settings = AppSettings()
    orchestrator = VoiceOrchestrator(settings)
    app = FastAPI(
        title="Voice Integration Gateway",
        version="0.2.0",
        description=(
            "Orchestrates STT -> RAG -> LLM -> TTS with optional "
            "NDJSON streaming output."
        ),
    )

    def payload(result):
        data = result.model_dump(mode="json")
        audio_path = result.audio_path
        data["audio_wav_base64"] = None
        if audio_path:
            data["audio_wav_base64"] = base64.b64encode(
                Path(audio_path).read_bytes()
            ).decode("ascii")
        return data

    def stream_line(event: dict) -> str:
        value = dict(event)
        audio = value.pop("audio", None)
        if isinstance(audio, (bytes, bytearray)):
            value["audio_wav_base64"] = base64.b64encode(
                bytes(audio)
            ).decode("ascii")
        return json.dumps(value, ensure_ascii=False) + "\n"

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        return orchestrator.health()

    @app.post("/v1/chat/text")
    def chat_text(request: TextChatRequest):
        try:
            result = orchestrator.process(
                query_text=request.query,
                synthesize=request.synthesize,
                rag_mode=request.rag_mode,
                rerank=request.rerank,
                top_k=request.top_k,
                voice_index=request.voice_index,
            )
            return payload(result)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            ) from exc

    @app.post("/v1/chat/text/stream")
    def chat_text_stream(request: TextChatRequest):
        def source():
            try:
                for event in orchestrator.stream_process(
                    query_text=request.query,
                    synthesize=request.synthesize,
                    rag_mode=request.rag_mode,
                    rerank=request.rerank,
                    top_k=request.top_k,
                    voice_index=request.voice_index,
                ):
                    yield stream_line(event)
            except Exception as exc:
                yield stream_line(
                    {
                        "type": "error",
                        "stage": "gateway",
                        "error": str(exc),
                    }
                )

        return StreamingResponse(
            iterate_in_threadpool(source()),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/chat/voice")
    async def chat_voice(
        audio: UploadFile = File(...),
        rag_mode: str | None = Form(default=None),
        rerank: bool | None = Form(default=None),
        top_k: int | None = Form(default=None),
        synthesize: bool = Form(default=True),
        voice_index: int | None = Form(default=None),
    ):
        suffix = Path(
            audio.filename or "audio.wav"
        ).suffix or ".wav"
        data = await audio.read()
        if not data:
            raise HTTPException(
                status_code=400,
                detail="Empty audio upload",
            )

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as handle:
                handle.write(data)
                temp_path = Path(handle.name)

            result = await asyncio.to_thread(
                orchestrator.process,
                audio_path=temp_path,
                synthesize=synthesize,
                rag_mode=rag_mode,
                rerank=rerank,
                top_k=top_k,
                voice_index=voice_index,
            )
            return payload(result)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            ) from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @app.post("/v1/chat/voice/stream")
    async def chat_voice_stream(
        audio: UploadFile = File(...),
        rag_mode: str | None = Form(default=None),
        rerank: bool | None = Form(default=None),
        top_k: int | None = Form(default=None),
        synthesize: bool = Form(default=True),
        voice_index: int | None = Form(default=None),
    ):
        suffix = Path(
            audio.filename or "audio.wav"
        ).suffix or ".wav"
        data = await audio.read()
        if not data:
            raise HTTPException(
                status_code=400,
                detail="Empty audio upload",
            )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as handle:
            handle.write(data)
            temp_path = Path(handle.name)

        def source():
            try:
                for event in orchestrator.stream_process(
                    audio_path=temp_path,
                    synthesize=synthesize,
                    rag_mode=rag_mode,
                    rerank=rerank,
                    top_k=top_k,
                    voice_index=voice_index,
                ):
                    yield stream_line(event)
            except Exception as exc:
                yield stream_line(
                    {
                        "type": "error",
                        "stage": "gateway",
                        "error": str(exc),
                    }
                )
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        return StreamingResponse(
            iterate_in_threadpool(source()),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Uvicorn missing. Run: pip install -e '.[api]'"
        ) from exc

    settings = AppSettings()
    uvicorn.run(
        "voice_app.gateway:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
