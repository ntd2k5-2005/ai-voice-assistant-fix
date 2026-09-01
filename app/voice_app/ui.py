from __future__ import annotations

import json
import traceback

from .config import AppSettings
from .errors import VoiceAppError
from .orchestrator import VoiceOrchestrator


def _knowledge_rows(documents: list[dict]) -> list[list]:
    rows = []
    for doc in documents:
        rows.append(
            [
                doc.get("document_id"),
                doc.get("filename"),
                doc.get("version"),
                doc.get("status"),
                doc.get("chunk_count"),
                doc.get("updated_at"),
                doc.get("cleanup_status"),
            ]
        )
    return rows


def _chunk_rows(payload: dict) -> list[list]:
    rows = []
    for chunk in payload.get("chunks") or []:
        text = str(chunk.get("text") or "").replace("\n", " ")
        rows.append(
            [
                chunk.get("chunk_id"),
                chunk.get("page"),
                chunk.get("section"),
                " > ".join(chunk.get("heading_path") or []),
                text[:500],
            ]
        )
    return rows


def build_demo(settings: AppSettings | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Run: pip install -e '.[demo]'"
        ) from exc

    settings = settings or AppSettings()
    orchestrator = VoiceOrchestrator(settings)

    def check_services():
        return orchestrator.health()

    def refresh_knowledge():
        try:
            docs = orchestrator.rag.list_documents()
            ids = [
                str(doc.get("document_id"))
                for doc in docs
                if doc.get("document_id")
            ]
            active_docs = [
                doc for doc in docs
                if str(doc.get("status") or "").lower() in {"active", "indexing"}
            ]
            inactive_count = len(docs) - len(active_docs)
            total_chunks = sum(
                int(doc.get("chunk_count") or 0)
                for doc in active_docs
            )
            summary = (
                f"**{len(active_docs)} tài liệu active** · "
                f"**{total_chunks} chunks**"
                + (
                    f" · {inactive_count} bản ghi lịch sử"
                    if inactive_count
                    else ""
                )
            )
            return (
                _knowledge_rows(docs),
                gr.update(
                    choices=ids,
                    value=ids[0] if ids else None,
                ),
                summary,
                "✅ Đã tải lại danh sách tri thức.",
            )
        except Exception as exc:
            return (
                [],
                gr.update(choices=[], value=None),
                "**0 tài liệu**",
                f"❌ {exc}",
            )

    def ingest_knowledge(
        file_path,
        document_id,
        metadata_json,
    ):
        if not file_path:
            rows, dropdown, summary, _ = refresh_knowledge()
            return (
                "⚠️ Chọn file trước khi nạp.",
                rows,
                dropdown,
                summary,
            )
        try:
            metadata = {}
            if (metadata_json or "").strip():
                metadata = json.loads(metadata_json)
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "Metadata phải là JSON object"
                    )

            result = orchestrator.rag.upload_document(
                file_path,
                document_id=(document_id or "").strip()
                or None,
                metadata=metadata,
            )
            rows, dropdown, summary, _ = refresh_knowledge()
            status = (
                "✅ Nạp thành công "
                f"**{result.get('document_id')}** · "
                f"{result.get('chunk_count', 0)} chunks · "
                f"version {result.get('version')}"
            )
            return status, rows, dropdown, summary
        except Exception as exc:
            rows, dropdown, summary, _ = refresh_knowledge()
            return (
                f"❌ Nạp tri thức thất bại: {exc}",
                rows,
                dropdown,
                summary,
            )

    def inspect_document(document_id):
        if not document_id:
            return (
                {},
                [],
                "Chọn tài liệu để xem chi tiết.",
            )
        try:
            payload = orchestrator.rag.get_document(
                document_id,
                include_chunks=True,
            )
            return (
                payload.get("document") or payload,
                _chunk_rows(payload),
                f"✅ Đã tải **{document_id}**.",
            )
        except Exception as exc:
            return {}, [], f"❌ {exc}"

    def delete_document(document_id, confirmed):
        rows, dropdown, summary, _ = refresh_knowledge()
        if not document_id:
            return (
                "⚠️ Chọn tài liệu cần xóa.",
                rows,
                dropdown,
                summary,
            )
        if not confirmed:
            return (
                "⚠️ Tick xác nhận trước khi xóa.",
                rows,
                dropdown,
                summary,
            )
        try:
            result = orchestrator.rag.delete_document(
                document_id
            )
            rows, dropdown, summary, _ = refresh_knowledge()
            return (
                f"✅ Đã xóa **{document_id}** "
                f"(`deleted={result.get('deleted')}`).",
                rows,
                dropdown,
                summary,
            )
        except Exception as exc:
            rows, dropdown, summary, _ = refresh_knowledge()
            return (
                f"❌ Xóa thất bại: {exc}",
                rows,
                dropdown,
                summary,
            )

    def reindex_document(
        document_id,
        file_path,
        metadata_json,
    ):
        rows, dropdown, summary, _ = refresh_knowledge()
        if not document_id or not file_path:
            return (
                "⚠️ Chọn document_id và file thay thế.",
                rows,
                dropdown,
                summary,
            )
        try:
            metadata = {}
            if (metadata_json or "").strip():
                metadata = json.loads(metadata_json)
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "Metadata phải là JSON object"
                    )
            result = orchestrator.rag.upload_document(
                file_path,
                document_id=document_id,
                metadata=metadata,
                reindex=True,
            )
            rows, dropdown, summary, _ = refresh_knowledge()
            return (
                f"✅ Reindex **{document_id}** → "
                f"version {result.get('version')} · "
                f"{result.get('chunk_count', 0)} chunks.",
                rows,
                dropdown,
                summary,
            )
        except Exception as exc:
            rows, dropdown, summary, _ = refresh_knowledge()
            return (
                f"❌ Reindex thất bại: {exc}",
                rows,
                dropdown,
                summary,
            )

    def run_streaming_turn(
        audio_path,
        text_input,
        rag_mode,
        rerank,
        top_k,
        synthesize,
        voice_index,
    ):
        transcript = ""
        answer = ""
        source_rows = []
        timings = {}
        details = {}

        try:
            # The realtime sink is mounted but CSS-hidden. The visible replay
            # player is cleared/hidden until TTS has completed the turn.
            yield (
                gr.skip(),
                "",
                None,
                gr.update(value=None, visible=False),
                [],
                {},
                {},
                "⏳ Đang xử lý STT/RAG…",
            )

            for event in orchestrator.stream_process(
                audio_path=audio_path,
                query_text=(text_input or "").strip() or None,
                synthesize=bool(synthesize),
                rag_mode=rag_mode,
                rerank=bool(rerank),
                top_k=int(top_k),
                voice_index=int(voice_index),
            ):
                kind = event["type"]

                if kind == "prepared":
                    transcript = event.get("transcript") or ""
                    source_rows = event.get("sources") or []
                    timings.update(event.get("timings") or {})
                    details.update(event.get("details") or {})
                    yield (
                        transcript,
                        "",
                        gr.skip(),
                        gr.skip(),
                        source_rows,
                        timings,
                        details,
                        "🧠 Đã truy xuất tri thức · LLM đang trả lời…",
                    )

                elif kind == "llm_delta":
                    answer = event.get("answer") or answer
                    yield (
                        gr.skip(),
                        answer,
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        "✍️ LLM đang stream · câu hoàn chỉnh sẽ vào hàng đợi TTS…",
                    )

                elif kind == "audio":
                    buffer_ms = event.get("estimated_buffer_ahead_ms")
                    buffer_note = (
                        f" · buffer≈{buffer_ms:.0f} ms"
                        if isinstance(buffer_ms, (int, float))
                        else ""
                    )
                    sentences = event.get("sentence_indices") or []
                    sentence_note = (
                        f"câu {','.join(str(x) for x in sentences)}"
                        if sentences
                        else "audio"
                    )
                    yield (
                        gr.skip(),
                        gr.skip(),
                        event["audio"],
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        "🔊 AI đang nói · "
                        f"{sentence_note} · native PCM stream + adaptive jitter"
                        f"{buffer_note}",
                    )

                elif kind == "tts_done":
                    tts_meta = event.get("meta") or {}
                    details["tts"] = tts_meta
                    final_audio = event.get("final_audio")
                    yield (
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.update(
                            value=final_audio,
                            visible=bool(final_audio),
                        ),
                        gr.skip(),
                        gr.skip(),
                        details,
                        "✅ TTS đã tổng hợp xong · "
                        "thanh nghe lại đã sẵn sàng.",
                    )

                elif kind == "error":
                    yield (
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        {"error": event},
                        f"❌ Lỗi {event.get('stage')}: {event.get('error')}",
                    )
                    return

                elif kind == "done":
                    answer = event.get("answer") or answer
                    timings = event.get("timings") or timings
                    details = event.get("details") or details
                    first_audio = timings.get("first_audio_ms")
                    extra = (
                        f" · first audio={first_audio:.0f} ms"
                        if isinstance(first_audio, (int, float))
                        else ""
                    )
                    tts = details.get("tts") or {}
                    underruns = tts.get("estimated_underrun_count")
                    queue_depth = tts.get("max_sentence_queue_depth")
                    version = tts.get("pipeline_version")
                    runtime_note = (
                        f" · {version or 'pipeline'} · underrun≈{underruns} · queue≤{queue_depth}"
                        if underruns is not None
                        else (f" · {version}" if version else "")
                    )
                    yield (
                        event.get("transcript") or transcript,
                        answer,
                        gr.skip(),
                        gr.skip(),
                        event.get("sources") or source_rows,
                        timings,
                        details,
                        "✅ Hoàn tất · "
                        f"total={timings.get('total_ms', 0):.0f} ms"
                        f"{extra}{runtime_note}",
                    )

        except VoiceAppError as exc:
            yield (
                transcript,
                answer,
                gr.skip(),
                gr.skip(),
                source_rows,
                timings,
                {"error": str(exc)},
                f"❌ Integration error: {exc}",
            )
        except Exception as exc:
            yield (
                transcript,
                answer,
                gr.skip(),
                gr.skip(),
                source_rows,
                timings,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                },
                f"❌ Unexpected error: {exc}",
            )

    css = """
    .gradio-container {
        max-width: 1500px !important;
        margin: 0 auto !important;
    }
    .hero {
        padding: 8px 4px 18px 4px;
    }
    .hero h1 {
        font-size: 2rem;
        margin-bottom: 4px;
    }
    .hero p {
        opacity: .72;
        margin-top: 0;
    }
    .pill {
        display: inline-block;
        padding: 5px 10px;
        border: 1px solid var(--border-color-primary);
        border-radius: 999px;
        margin-right: 6px;
        font-size: .85rem;
    }
    .panel {
        border: 1px solid var(--border-color-primary);
        border-radius: 16px;
        padding: 10px;
    }
    .status-card {
        border-left: 4px solid var(--color-accent);
        padding-left: 12px;
    }
    .knowledge-summary {
        font-size: 1.05rem;
    }
    #realtime-audio-sink {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    """

    with gr.Blocks(
        title="HUIT Voice Knowledge Assistant",
        css=css,
        fill_width=True,
    ) as demo:
        gr.HTML(
            """
            <div class="hero">
                <h1>🎙️ HUIT Voice Knowledge Assistant</h1>
                <p>
                    Trợ lý giọng nói RAG · LLM token stream · sentence-first native PCM ·
                    adaptive jitter buffer V7 · quản lý kho tri thức trực tiếp
                </p>
                <span class="pill">STT</span>
                <span class="pill">RAG</span>
                <span class="pill">LLM streaming</span>
                <span class="pill">TTS native stream V7</span>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("💬 Trợ lý"):
                with gr.Row(equal_height=False):
                    with gr.Column(
                        scale=5,
                        elem_classes=["panel"],
                    ):
                        audio = gr.Audio(
                            sources=[
                                "microphone",
                                "upload",
                            ],
                            type="filepath",
                            label=(
                                "🎤 Nói câu hỏi — dừng ghi âm "
                                "là hệ thống tự chạy"
                            ),
                        )
                        text_input = gr.Textbox(
                            label="Hoặc nhập câu hỏi",
                            placeholder=(
                                "Nhập câu hỏi rồi nhấn Enter…"
                            ),
                            lines=2,
                        )
                        send_button = gr.Button(
                            "Gửi câu hỏi",
                            variant="primary",
                        )
                        status = gr.Markdown(
                            "Sẵn sàng. Ghi âm hoặc tải audio "
                            "lên để tự chạy.",
                            elem_classes=["status-card"],
                        )

                    with gr.Column(
                        scale=7,
                        elem_classes=["panel"],
                    ):
                        transcript = gr.Textbox(
                            label="📝 Transcript",
                            interactive=False,
                            lines=2,
                        )
                        answer = gr.Markdown(
                            label="🧠 Câu trả lời"
                        )
                        # Realtime sink stays mounted for Gradio/HLS autoplay,
                        # but CSS hides its player UI. The user only sees a
                        # replay control after TTS completes.
                        realtime_audio = gr.Audio(
                            label="Realtime audio sink",
                            streaming=True,
                            autoplay=True,
                            format="wav",
                            interactive=False,
                            show_download_button=False,
                            elem_id="realtime-audio-sink",
                        )
                        final_audio = gr.Audio(
                            label="🔁 Nghe lại câu trả lời",
                            streaming=False,
                            autoplay=False,
                            format="wav",
                            interactive=False,
                            visible=False,
                        )

                with gr.Accordion(
                    "⚙️ Cấu hình truy vấn",
                    open=False,
                ):
                    with gr.Row():
                        rag_mode = gr.Dropdown(
                            [
                                "dense",
                                "sparse",
                                "hybrid",
                            ],
                            value=settings.rag_mode,
                            label="RAG mode",
                        )
                        rerank = gr.Checkbox(
                            value=settings.rag_rerank,
                            label="Rerank",
                        )
                        top_k = gr.Slider(
                            1,
                            12,
                            value=settings.rag_top_k,
                            step=1,
                            label="Top-k contexts",
                        )
                        synthesize = gr.Checkbox(
                            value=True,
                            label="Tự phát TTS",
                        )
                        gr.Markdown(
                            f"TTS V7 mode: **{settings.tts_generation_mode}** · "
                            f"startup buffer: **{settings.tts_startup_buffer_ms} ms** · "
                            f"queue: **{settings.tts_sentence_queue_max}/{settings.tts_audio_queue_max}**"
                        )
                        voice_index = gr.Number(
                            value=settings.tts_voice_index,
                            precision=0,
                            label="TTS voice",
                        )

                with gr.Accordion(
                    "🔎 Nguồn & hiệu năng",
                    open=False,
                ):
                    with gr.Row():
                        sources = gr.JSON(
                            label="Retrieved sources"
                        )
                        timings = gr.JSON(
                            label=(
                                "Latency / streaming metrics"
                            )
                        )
                    service_details = gr.JSON(
                        label="Raw integration details"
                    )

            with gr.Tab("📚 Kho tri thức"):
                with gr.Row():
                    with gr.Column(
                        scale=4,
                        elem_classes=["panel"],
                    ):
                        gr.Markdown("### Nạp tài liệu")
                        knowledge_file = gr.File(
                            label="Tài liệu",
                            type="filepath",
                        )
                        knowledge_id = gr.Textbox(
                            label=(
                                "Document ID "
                                "(để trống = tên file)"
                            )
                        )
                        knowledge_metadata = gr.Textbox(
                            label="Metadata JSON",
                            value=(
                                '{"source": "manual-upload"}'
                            ),
                            lines=5,
                        )
                        ingest_button = gr.Button(
                            "⬆️ Nạp vào kho tri thức",
                            variant="primary",
                        )
                        ingest_status = gr.Markdown()

                    with gr.Column(
                        scale=8,
                        elem_classes=["panel"],
                    ):
                        with gr.Row():
                            knowledge_summary = gr.Markdown(
                                "**0 tài liệu**",
                                elem_classes=[
                                    "knowledge-summary"
                                ],
                            )
                            refresh_button = gr.Button(
                                "↻ Làm mới",
                                size="sm",
                            )
                        knowledge_table = gr.Dataframe(
                            headers=[
                                "Document ID",
                                "Filename",
                                "Version",
                                "Status",
                                "Chunks",
                                "Updated",
                                "Cleanup",
                            ],
                            datatype=[
                                "str",
                                "str",
                                "number",
                                "str",
                                "number",
                                "str",
                                "str",
                            ],
                            interactive=False,
                            wrap=True,
                            label="Danh sách tài liệu",
                        )

                gr.Markdown("### Quản lý tài liệu")
                with gr.Row():
                    with gr.Column(scale=4):
                        doc_selector = gr.Dropdown(
                            [],
                            label="Chọn document",
                        )
                        inspect_button = gr.Button(
                            "Xem chi tiết"
                        )
                        delete_confirm = gr.Checkbox(
                            label=(
                                "Tôi xác nhận xóa "
                                "document này"
                            )
                        )
                        delete_button = gr.Button(
                            "🗑️ Xóa",
                            variant="stop",
                        )
                        manage_status = gr.Markdown()

                    with gr.Column(scale=8):
                        document_detail = gr.JSON(
                            label="Document metadata"
                        )
                        chunks_table = gr.Dataframe(
                            headers=[
                                "Chunk ID",
                                "Page",
                                "Section",
                                "Heading path",
                                "Excerpt",
                            ],
                            datatype=[
                                "str",
                                "number",
                                "str",
                                "str",
                                "str",
                            ],
                            interactive=False,
                            wrap=True,
                            label="Chunks",
                        )

                with gr.Accordion(
                    "♻️ Reindex / thay thế phiên bản",
                    open=False,
                ):
                    reindex_file = gr.File(
                        label="File thay thế",
                        type="filepath",
                    )
                    reindex_metadata = gr.Textbox(
                        label="Metadata JSON",
                        value=(
                            '{"source": "manual-reindex"}'
                        ),
                        lines=4,
                    )
                    reindex_button = gr.Button(
                        "Reindex document đang chọn"
                    )

            with gr.Tab("🩺 Hệ thống"):
                gr.Markdown("### Trạng thái dịch vụ")
                health_button = gr.Button(
                    "Kiểm tra STT · RAG · LLM · TTS",
                    variant="primary",
                )
                health = gr.JSON(label="Service health")
                gr.Markdown(
                    "LLM endpoint streaming: `/v1/chat/stream` · "
                    "V7 realtime audio chạy qua sink ẩn; "
                    "player nghe lại chỉ hiện sau khi TTS hoàn tất."
                )

        stream_inputs = [
            audio,
            text_input,
            rag_mode,
            rerank,
            top_k,
            synthesize,
            voice_index,
        ]
        stream_outputs = [
            transcript,
            answer,
            realtime_audio,
            final_audio,
            sources,
            timings,
            service_details,
            status,
        ]

        if settings.auto_run_voice:
            audio.stop_recording(
                fn=run_streaming_turn,
                inputs=stream_inputs,
                outputs=stream_outputs,
                show_progress="minimal",
                concurrency_limit=1,
            )
        if settings.auto_run_upload:
            audio.upload(
                fn=run_streaming_turn,
                inputs=stream_inputs,
                outputs=stream_outputs,
                show_progress="minimal",
                concurrency_limit=1,
            )
        text_input.submit(
            fn=run_streaming_turn,
            inputs=stream_inputs,
            outputs=stream_outputs,
            show_progress="minimal",
            concurrency_limit=1,
        )
        send_button.click(
            fn=run_streaming_turn,
            inputs=stream_inputs,
            outputs=stream_outputs,
            show_progress="minimal",
            concurrency_limit=1,
        )

        health_button.click(
            fn=check_services,
            outputs=health,
        )
        refresh_button.click(
            fn=refresh_knowledge,
            outputs=[
                knowledge_table,
                doc_selector,
                knowledge_summary,
                manage_status,
            ],
        )
        ingest_button.click(
            fn=ingest_knowledge,
            inputs=[
                knowledge_file,
                knowledge_id,
                knowledge_metadata,
            ],
            outputs=[
                ingest_status,
                knowledge_table,
                doc_selector,
                knowledge_summary,
            ],
        )
        inspect_button.click(
            fn=inspect_document,
            inputs=doc_selector,
            outputs=[
                document_detail,
                chunks_table,
                manage_status,
            ],
        )
        doc_selector.change(
            fn=inspect_document,
            inputs=doc_selector,
            outputs=[
                document_detail,
                chunks_table,
                manage_status,
            ],
        )
        delete_button.click(
            fn=delete_document,
            inputs=[
                doc_selector,
                delete_confirm,
            ],
            outputs=[
                manage_status,
                knowledge_table,
                doc_selector,
                knowledge_summary,
            ],
        )
        reindex_button.click(
            fn=reindex_document,
            inputs=[
                doc_selector,
                reindex_file,
                reindex_metadata,
            ],
            outputs=[
                manage_status,
                knowledge_table,
                doc_selector,
                knowledge_summary,
            ],
        )
        demo.load(
            fn=refresh_knowledge,
            outputs=[
                knowledge_table,
                doc_selector,
                knowledge_summary,
                manage_status,
            ],
        )

        demo.queue(
            default_concurrency_limit=1,
            max_size=20,
        )

    return demo


def main() -> None:
    settings = AppSettings()
    demo = build_demo(settings)
    demo.launch(
        server_name=settings.demo_host,
        server_port=settings.demo_port,
        share=settings.demo_share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
