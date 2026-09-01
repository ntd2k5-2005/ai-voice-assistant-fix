from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .clients import LLMClient, RAGClient, STTClient, TTSClient
from .config import AppSettings
from .contracts import (
    ChatMessage,
    PipelineResult,
    PipelineTimings,
    SourceView,
)
from .prompt_builder import PromptBuilder
from .realtime import (
    AdaptiveJitterController,
    AdaptiveStreamJitterController,
    AudioPostProcessor,
    ContinuousPCMBuffer,
    PCMDispatch,
    PCMStreamChunk,
    PreparedSentenceAudio,
    SentenceScheduler,
    TTSPrefetchWorker,
    TTSStreamWorker,
)
from .streaming import SentenceChunker, pcm_to_wav_bytes


class VoiceOrchestrator:
    """Coordinates STT -> RAG -> LLM/TTS for blocking and streaming flows."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        stt_client=None,
        rag_client=None,
        llm_client=None,
        tts_client=None,
    ) -> None:
        self.settings = settings or AppSettings()

        self.stt = stt_client or STTClient(
            self.settings.stt_base_url,
            timeout_s=self.settings.stt_timeout_s,
        )
        self.rag = rag_client or RAGClient(
            self.settings.rag_base_url,
            timeout_s=self.settings.rag_timeout_s,
            require_rerank_applied=(
                self.settings.rag_require_rerank_applied
            ),
        )
        self.llm = llm_client or LLMClient(
            self.settings.llm_base_url,
            timeout_s=self.settings.llm_timeout_s,
        )
        self.tts = tts_client or TTSClient(
            self.settings.tts_base_url,
            timeout_s=self.settings.tts_timeout_s,
            sample_rate=self.settings.tts_sample_rate,
            channels=self.settings.tts_channels,
            dtype=self.settings.tts_dtype,
            stream_chunk_ms=self.settings.tts_stream_chunk_ms,
            generation_mode=self.settings.tts_generation_mode,
            reject_all_silence=(
                self.settings.tts_reject_all_silence
            ),
            output_dir=self.settings.tts_output_dir,
            keep_last_files=self.settings.tts_keep_last_files,
        )

        self.prompt_builder = PromptBuilder(
            system_prompt=self.settings.system_prompt,
            max_context_chars_per_item=(
                self.settings.max_context_chars_per_item
            ),
            max_history_turns=self.settings.max_history_turns,
        )

    def close(self) -> None:
        for client in (
            self.stt,
            self.rag,
            self.llm,
            self.tts,
        ):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def health(self) -> dict[str, Any]:
        statuses = {
            "stt": self.stt.health(),
            "rag": self.rag.health(),
            "llm": self.llm.health(),
            "tts": self.tts.health(),
        }
        reachable = all(
            status.reachable for status in statuses.values()
        )
        known_not_ready = any(
            status.ready is False
            for status in statuses.values()
        )
        unknown_ready = any(
            status.ready is None
            for status in statuses.values()
        )
        return {
            "reachable": reachable,
            "ready": (
                reachable
                and not known_not_ready
                and not unknown_ready
            ),
            "note": (
                "All four service readiness contracts are "
                "checked through their APIs."
            ),
            "services": {
                name: status.model_dump(mode="json")
                for name, status in statuses.items()
            },
        }

    def _sources(self, rag_result) -> list[SourceView]:
        limit = self.settings.source_excerpt_chars
        views: list[SourceView] = []
        for ctx in rag_result.contexts:
            excerpt = ctx.text.strip()
            if len(excerpt) > limit:
                excerpt = excerpt[:limit].rstrip() + "…"
            views.append(
                SourceView(
                    rank=ctx.rank,
                    document_id=ctx.document_id,
                    chunk_id=ctx.chunk_id,
                    score=ctx.score,
                    score_source=ctx.score_source,
                    source=ctx.source,
                    section=ctx.section,
                    page=ctx.page,
                    heading_path=ctx.heading_path,
                    excerpt=excerpt,
                )
            )
        return views

    def _prepare(
        self,
        *,
        audio_path,
        query_text,
        history,
        rag_mode,
        rerank,
        top_k,
    ):
        stt_result = None
        transcript = (query_text or "").strip()

        if not transcript:
            if audio_path is None:
                raise ValueError(
                    "Provide audio_path or query_text"
                )
            stt_result = self.stt.transcribe(audio_path)
            transcript = stt_result.text.strip()

        if not transcript:
            raise ValueError("STT returned an empty transcript")

        mode = (
            rag_mode or self.settings.rag_mode
        ).strip().lower()
        rerank_value = (
            self.settings.rag_rerank
            if rerank is None
            else bool(rerank)
        )
        top_k_value = top_k or self.settings.rag_top_k

        rag_result = self.rag.retrieve(
            transcript,
            mode=mode,
            rerank=rerank_value,
            top_k=top_k_value,
            candidate_k=max(
                self.settings.rag_candidate_k,
                top_k_value,
            ),
            debug=False,
        )

        messages = self.prompt_builder.build(
            question=transcript,
            contexts=rag_result.contexts,
            history=history,
        )
        return (
            stt_result,
            transcript,
            rag_result,
            messages,
        )

    def process(
        self,
        *,
        audio_path: str | Path | None = None,
        query_text: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        synthesize: bool = True,
        rag_mode: str | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
        voice_index: int | None = None,
    ) -> PipelineResult:
        """Compatibility blocking path used by gateway and regression tests."""
        started_total = time.perf_counter()
        (
            stt_result,
            transcript,
            rag_result,
            messages,
        ) = self._prepare(
            audio_path=audio_path,
            query_text=query_text,
            history=history,
            rag_mode=rag_mode,
            rerank=rerank,
            top_k=top_k,
        )

        llm_result = self.llm.chat(
            messages,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
            max_tokens=self.settings.llm_max_tokens,
        )

        tts_result = None
        if synthesize:
            tts_result = self.tts.synthesize(
                llm_result.text,
                voice_index=(
                    self.settings.tts_voice_index
                    if voice_index is None
                    else int(voice_index)
                ),
            )

        timings = PipelineTimings(
            stt_ms=(
                stt_result.client_latency_ms
                if stt_result
                else 0.0
            ),
            rag_ms=rag_result.client_latency_ms,
            llm_ms=llm_result.client_latency_ms,
            tts_ms=(
                tts_result.total_ms
                if tts_result
                else 0.0
            ),
            total_ms=(
                time.perf_counter() - started_total
            ) * 1000,
            rag_internal_ms=rag_result.latency.total_ms,
            stt_inference_ms=(
                stt_result.inference_ms
                if stt_result
                else None
            ),
            tts_ttfb_ms=(
                tts_result.ttfb_ms
                if tts_result
                else None
            ),
        )

        return PipelineResult(
            transcript=transcript,
            answer=llm_result.text,
            audio_path=(
                tts_result.audio_path
                if tts_result
                else None
            ),
            sources=self._sources(rag_result),
            timings=timings,
            rag_mode=rag_result.mode,
            rerank_applied=rag_result.rerank_applied,
            llm_model=llm_result.model,
            service_details={
                "stt": (
                    stt_result.model_dump(mode="json")
                    if stt_result
                    else {"skipped": True}
                ),
                "rag": {
                    "mode": rag_result.mode,
                    "rerank_requested": (
                        rag_result.rerank_requested
                    ),
                    "rerank_applied": (
                        rag_result.rerank_applied
                    ),
                    "context_count": len(
                        rag_result.contexts
                    ),
                    "latency": (
                        rag_result.latency.model_dump(
                            mode="json"
                        )
                    ),
                },
                "llm": {
                    "model": llm_result.model,
                    "request_id": llm_result.request_id,
                    "client_latency_ms": (
                        llm_result.client_latency_ms
                    ),
                },
                "tts": (
                    tts_result.model_dump(mode="json")
                    if tts_result
                    else {"skipped": True}
                ),
            },
        )

    def _stream_process_v6(
        self,
        *,
        audio_path: str | Path | None = None,
        query_text: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        synthesize: bool = True,
        rag_mode: str | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
        voice_index: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """V6 realtime turn: sentence scheduling -> TTS prefetch -> jittered playout.

        The key change from V5 is that browser playback no longer consumes
        VieNeu's native frame stream directly. Complete semantic sentences are
        synthesized into full PCM segments, queued, and prefetched. The next
        sentence can therefore synthesize while an earlier sentence is already
        audible. A bounded sentence queue, bounded audio queue, adaptive startup
        prebuffer and server-side playout estimate provide backpressure and
        reduce the stop-start behaviour seen in V5.
        """
        started_total = time.perf_counter()
        (
            stt_result,
            transcript,
            rag_result,
            messages,
        ) = self._prepare(
            audio_path=audio_path,
            query_text=query_text,
            history=history,
            rag_mode=rag_mode,
            rerank=rerank,
            top_k=top_k,
        )

        source_views = [
            source.model_dump(mode="json")
            for source in self._sources(rag_result)
        ]
        base_details = {
            "stt": (
                stt_result.model_dump(mode="json")
                if stt_result
                else {"skipped": True}
            ),
            "rag": {
                "mode": rag_result.mode,
                "rerank_requested": rag_result.rerank_requested,
                "rerank_applied": rag_result.rerank_applied,
                "context_count": len(rag_result.contexts),
                "latency": rag_result.latency.model_dump(mode="json"),
            },
        }

        yield {
            "type": "prepared",
            "transcript": transcript,
            "sources": source_views,
            "timings": {
                "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
                "stt_inference_ms": stt_result.inference_ms if stt_result else None,
                "rag_ms": rag_result.client_latency_ms,
                "rag_internal_ms": rag_result.latency.total_ms,
            },
            "details": base_details,
        }

        control_events: queue.Queue[dict[str, Any]] = queue.Queue()
        audio_queue: queue.Queue[Any] = queue.Queue(
            maxsize=self.settings.tts_audio_queue_max
        )
        cancel_event = threading.Event()
        llm_done = threading.Event()
        tts_done = threading.Event()
        answer_parts: list[str] = []

        sentences = SentenceScheduler(
            maxsize=self.settings.tts_sentence_queue_max,
            started_at=started_total,
            cancel_event=cancel_event,
        )

        voice = (
            self.settings.tts_voice_index
            if voice_index is None
            else int(voice_index)
        )

        postprocessor = AudioPostProcessor(
            sample_rate=self.settings.tts_sample_rate,
            channels=self.settings.tts_channels,
            dtype=self.settings.tts_dtype,
            enabled=self.settings.tts_trim_silence,
            silence_threshold=self.settings.tts_silence_threshold,
            leading_keep_ms=self.settings.tts_leading_silence_ms,
            trailing_keep_ms=self.settings.tts_trailing_silence_ms,
            sentence_pause_ms=self.settings.tts_inter_sentence_pause_ms,
        )

        tts_prefetch = TTSPrefetchWorker(
            tts_client=self.tts,
            voice_index=voice,
            generation_mode=self.settings.tts_generation_mode,
            postprocessor=postprocessor,
            audio_queue=audio_queue,
            started_at=started_total,
        )

        jitter = AdaptiveJitterController(
            started_at=started_total,
            base_startup_ms=self.settings.tts_startup_buffer_ms,
            min_startup_ms=self.settings.tts_startup_buffer_min_ms,
            max_startup_ms=self.settings.tts_startup_buffer_max_ms,
            startup_max_wait_ms=self.settings.tts_startup_max_wait_ms,
            max_buffer_ahead_ms=self.settings.tts_max_buffer_ahead_ms,
        )

        def llm_worker() -> None:
            chunker = SentenceChunker(
                min_chars=self.settings.tts_phrase_min_chars,
                soft_chars=self.settings.tts_phrase_soft_chars,
                hard_chars=self.settings.tts_phrase_hard_chars,
            )
            first_client_ttft: float | None = None
            try:
                for event in self.llm.chat_stream(
                    messages,
                    temperature=self.settings.llm_temperature,
                    top_p=self.settings.llm_top_p,
                    max_tokens=self.settings.llm_max_tokens,
                ):
                    if event["type"] == "delta":
                        delta = str(event.get("content") or "")
                        if first_client_ttft is None:
                            first_client_ttft = event.get("client_ttft_ms")
                        answer_parts.append(delta)
                        control_events.put(
                            {
                                "type": "llm_delta",
                                "delta": delta,
                                "answer": "".join(answer_parts),
                                "meta": event,
                            }
                        )
                        if synthesize:
                            for sentence in chunker.push(delta):
                                sentences.put(sentence)
                    elif event["type"] == "done":
                        if synthesize:
                            for sentence in chunker.flush():
                                sentences.put(sentence)
                        event["client_ttft_ms"] = first_client_ttft
                        event["tts_sentence_count"] = sentences.sentence_count
                        event["tts_phrase_count"] = sentences.sentence_count
                        control_events.put(
                            {
                                "type": "llm_done",
                                "answer": "".join(answer_parts).strip(),
                                "meta": event,
                            }
                        )
            except Exception as exc:
                control_events.put(
                    {"type": "error", "stage": "llm", "error": str(exc)}
                )
                cancel_event.set()
            finally:
                llm_done.set()
                if synthesize:
                    sentences.close()

        def tts_worker() -> None:
            if not synthesize:
                audio_queue.put(None)
                tts_done.set()
                return
            tts_prefetch.run(sentences)
            if tts_prefetch.error is not None:
                cancel_event.set()
            tts_done.set()

        threading.Thread(
            target=llm_worker,
            name="llm-stream-producer-v6",
            daemon=True,
        ).start()
        threading.Thread(
            target=tts_worker,
            name="tts-sentence-prefetch-v6",
            daemon=True,
        ).start()

        llm_meta: dict[str, Any] = {}
        had_error = False
        audio_source_done = not synthesize
        pending_audio: list[PreparedSentenceAudio] = []
        all_prepared: list[PreparedSentenceAudio] = []
        dispatch_records: list[dict[str, Any]] = []
        final_pcm_parts: list[bytes] = []
        first_audio_ready_ms: float | None = None
        first_tts_body_ms: float | None = None
        tts_started_at = time.perf_counter()

        def batch_ready(force: bool = False) -> bool:
            if not pending_audio:
                return False
            if force:
                return True
            total = sum(item.duration_ms for item in pending_audio)
            return total >= self.settings.tts_playback_chunk_ms

        def pop_batch(force: bool = False) -> list[PreparedSentenceAudio]:
            if not pending_audio:
                return []
            selected: list[PreparedSentenceAudio] = []
            total = 0.0
            while pending_audio:
                item = pending_audio[0]
                selected.append(pending_audio.pop(0))
                total += item.duration_ms
                if total >= self.settings.tts_playback_chunk_ms:
                    break
                if not force and not pending_audio:
                    break
            return selected

        def make_audio_event(batch: list[PreparedSentenceAudio]) -> dict[str, Any]:
            raw = b"".join(item.raw for item in batch)
            duration_ms = sum(item.duration_ms for item in batch)
            estimated_gap_ms, buffer_ahead_ms = jitter.record_dispatch(duration_ms)
            dispatch_ms = (time.perf_counter() - started_total) * 1000.0
            sentence_indices = [item.index for item in batch]
            dispatch_records.append(
                {
                    "dispatch_ms": dispatch_ms,
                    "duration_ms": duration_ms,
                    "sentence_indices": sentence_indices,
                }
            )
            return {
                "type": "audio",
                "audio": pcm_to_wav_bytes(
                    raw,
                    dtype=self.settings.tts_dtype,
                    channels=self.settings.tts_channels,
                    sample_rate=self.settings.tts_sample_rate,
                ),
                "sentence": " ".join(item.sentence for item in batch),
                "sentences": [item.sentence for item in batch],
                "sentence_indices": sentence_indices,
                "audio_chunk": len(dispatch_records),
                "duration_ms": duration_ms,
                "generation_mode": batch[0].generation_mode,
                "streaming_mode": batch[0].streaming_mode,
                "upstream_backend": batch[0].upstream_backend,
                "playout_strategy": "sentence-prefetch-adaptive-jitter",
                "estimated_server_gap_ms": estimated_gap_ms,
                "estimated_buffer_ahead_ms": buffer_ahead_ms,
                "startup_buffer_target_ms": jitter.metrics.startup_target_ms,
                "sentence_queue_depth": sentences.qsize(),
                "audio_queue_depth": audio_queue.qsize(),
            }

        while True:
            progressed = False

            # Keep text/UI events responsive even while audio prefetch is busy.
            for _ in range(32):
                try:
                    event = control_events.get_nowait()
                except queue.Empty:
                    break
                progressed = True
                if event["type"] == "llm_done":
                    llm_meta = event.get("meta") or {}
                elif event["type"] == "error":
                    had_error = True
                    cancel_event.set()
                yield event

            pending_ms = sum(item.duration_ms for item in pending_audio)
            estimated_ahead = jitter.buffer_ahead_ms()
            can_pull_audio = (
                jitter.metrics.first_dispatch_ms is None
                or (pending_ms + estimated_ahead) < self.settings.tts_max_buffer_ahead_ms
                or tts_done.is_set()
            )

            if not audio_source_done and can_pull_audio:
                try:
                    audio_item = audio_queue.get(timeout=0.01 if not progressed else 0.0)
                    progressed = True
                except queue.Empty:
                    audio_item = "__empty__"

                if audio_item is None:
                    audio_source_done = True
                elif isinstance(audio_item, dict) and audio_item.get("type") == "error":
                    had_error = True
                    cancel_event.set()
                    yield audio_item
                elif audio_item != "__empty__":
                    assert isinstance(audio_item, PreparedSentenceAudio)
                    pending_audio.append(audio_item)
                    all_prepared.append(audio_item)
                    final_pcm_parts.append(audio_item.raw)
                    if first_audio_ready_ms is None:
                        first_audio_ready_ms = audio_item.synthesis_finished_ms
                        if audio_item.ttfb_ms is not None:
                            first_tts_body_ms = (
                                audio_item.synthesis_started_ms + audio_item.ttfb_ms
                            )

            if pending_audio and jitter.metrics.first_dispatch_ms is None:
                buffered_ms = sum(item.duration_ms for item in pending_audio)
                first_rtf = pending_audio[0].rtf
                source_done = audio_source_done
                if jitter.startup_ready(
                    buffered_ms=buffered_ms,
                    first_rtf=first_rtf,
                    source_done=source_done,
                    prepared_segments=len(pending_audio),
                ):
                    # Once the adaptive startup controller releases playback,
                    # dispatch what is already prepared even when a short first
                    # sentence has not reached the normal HLS coalescing target.
                    # This keeps TTS_STARTUP_MAX_WAIT_MS a real upper bound
                    # instead of accidentally waiting for sentence B forever.
                    batch = pop_batch(force=True)
                    yield make_audio_event(batch)
                    progressed = True

            elif pending_audio and jitter.metrics.first_dispatch_ms is not None:
                # Coalesce short sentences to keep browser/HLS media chunks at
                # roughly one second or longer. Longer sentences remain intact.
                force = audio_source_done
                if batch_ready(force=force):
                    if (
                        jitter.buffer_ahead_ms() <= self.settings.tts_max_buffer_ahead_ms
                        or tts_done.is_set()
                    ):
                        batch = pop_batch(force=force)
                        yield make_audio_event(batch)
                        progressed = True

            finished = (
                llm_done.is_set()
                and tts_done.is_set()
                and audio_source_done
                and not pending_audio
                and control_events.empty()
            )
            if finished:
                break

            if not progressed:
                time.sleep(0.01)

        if had_error:
            return

        final_audio_path: str | None = None
        if synthesize and final_pcm_parts:
            raw_final = b"".join(final_pcm_parts)
            save_fn = getattr(self.tts, "save_pcm_wav", None)
            if callable(save_fn):
                final_audio_path = save_fn(raw_final, prefix="v6-final")
            else:
                import uuid

                output_dir = Path(self.settings.tts_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                final_path = output_dir / f"tts-v6-final-{uuid.uuid4().hex}.wav"
                final_path.write_bytes(
                    pcm_to_wav_bytes(
                        raw_final,
                        dtype=self.settings.tts_dtype,
                        channels=self.settings.tts_channels,
                        sample_rate=self.settings.tts_sample_rate,
                    )
                )
                final_audio_path = str(final_path)

        segment_metrics = [
            {
                "sentence": item.index,
                "chars": len(item.sentence),
                "sentence_text": item.sentence,
                "queued_ms": item.queued_ms,
                "sentence_wait_ms": item.sentence_wait_ms,
                "synthesis_started_ms": item.synthesis_started_ms,
                "synthesis_finished_ms": item.synthesis_finished_ms,
                "synthesis_ms": item.synthesis_ms,
                "audio_duration_ms": item.duration_ms,
                "original_audio_duration_ms": item.original_duration_ms,
                "trimmed_silence_ms": item.trimmed_ms,
                "appended_pause_ms": item.appended_pause_ms,
                "ttfb_ms": item.ttfb_ms,
                "rtf": item.rtf,
                "generation_mode": item.generation_mode,
                "streaming_mode": item.streaming_mode,
                "upstream_backend": item.upstream_backend,
                "sentence_queue_depth_at_enqueue": item.sentence_queue_depth_at_enqueue,
                "sentence_queue_depth_at_start": item.sentence_queue_depth_at_start,
            }
            for item in all_prepared
        ]

        rtfs = [item.rtf for item in all_prepared if item.rtf is not None]
        total_trimmed_ms = sum(item.trimmed_ms for item in all_prepared)
        total_audio_ms = sum(item.duration_ms for item in all_prepared)

        sentence_dispatch_ms: dict[int, float] = {}
        for record in dispatch_records:
            for index in record["sentence_indices"]:
                sentence_dispatch_ms[index] = record["dispatch_ms"]

        prefetch_overlap_count = 0
        prefetch_ready_before_previous_end_count = 0
        for current, previous in zip(all_prepared[1:], all_prepared[:-1]):
            previous_dispatch = sentence_dispatch_ms.get(previous.index)
            if previous_dispatch is None:
                continue
            estimated_previous_end = previous_dispatch + previous.duration_ms
            if current.synthesis_started_ms < estimated_previous_end:
                prefetch_overlap_count += 1
            if current.synthesis_finished_ms < estimated_previous_end:
                prefetch_ready_before_previous_end_count += 1

        tts_orchestration_ms = (time.perf_counter() - tts_started_at) * 1000.0
        generation_total_ms = (
            max(item.synthesis_finished_ms for item in all_prepared)
            - min(item.synthesis_started_ms for item in all_prepared)
            if all_prepared
            else 0.0
        )
        streaming_modes = sorted({item.streaming_mode for item in all_prepared})
        upstream_backends = sorted({item.upstream_backend for item in all_prepared})
        generation_modes = sorted({item.generation_mode for item in all_prepared})
        tts_meta = {
            "pipeline_version": "v6",
            "segmentation_mode": "sentence-first",
            "generation_strategy": "sentence-prefetch",
            "generation_mode": self.settings.tts_generation_mode,
            "generation_modes": generation_modes,
            "streaming_modes": streaming_modes,
            "upstream_backends": upstream_backends,
            "native_frame_streaming": any(
                mode == "frame-pcm" for mode in streaming_modes
            ),
            "playout_strategy": "sentence-prefetch-adaptive-jitter",
            "sentence_count": len(all_prepared),
            "segment_count": len(all_prepared),
            "playback_chunk_count": len(dispatch_records),
            "playback_audio_ms": total_audio_ms,
            "sentence_queue_capacity": self.settings.tts_sentence_queue_max,
            "max_sentence_queue_depth": sentences.max_depth,
            "sentence_backpressure_ms": sentences.total_backpressure_ms,
            "audio_queue_capacity": self.settings.tts_audio_queue_max,
            "max_audio_queue_depth": tts_prefetch.max_audio_queue_depth,
            "audio_queue_backpressure_ms": tts_prefetch.total_audio_queue_backpressure_ms,
            "startup_buffer_target_ms": jitter.metrics.startup_target_ms,
            "startup_buffer_observed_ms": jitter.metrics.startup_buffer_ms,
            "max_buffer_ahead_ms": jitter.metrics.max_buffer_ahead_ms,
            "estimated_underrun_count": jitter.metrics.estimated_underrun_count,
            "estimated_max_gap_ms": jitter.metrics.estimated_max_gap_ms,
            "prefetch_overlap_count": prefetch_overlap_count,
            "prefetch_overlap_ratio": (
                prefetch_overlap_count / max(1, len(all_prepared) - 1)
                if len(all_prepared) > 1
                else 0.0
            ),
            # Stronger look-ahead signal: sentence B is fully synthesized and
            # queue-ready before sentence A's estimated playback end. This is
            # the exact behavior V6 is trying to create.
            "prefetch_ready_before_previous_end_count": (
                prefetch_ready_before_previous_end_count
            ),
            "prefetch_ready_before_previous_end_ratio": (
                prefetch_ready_before_previous_end_count
                / max(1, len(all_prepared) - 1)
                if len(all_prepared) > 1
                else 0.0
            ),
            "pipeline_first_tts_body_ms": first_tts_body_ms,
            "pipeline_first_audio_ready_ms": first_audio_ready_ms,
            "pipeline_first_audio_ms": jitter.metrics.first_dispatch_ms,
            "mean_rtf": (sum(rtfs) / len(rtfs)) if rtfs else None,
            "max_rtf": max(rtfs) if rtfs else None,
            "trimmed_silence_ms": total_trimmed_ms,
            "final_audio_path": final_audio_path,
            # Preserve tts_ms semantics as synthesis/generation time. The
            # orchestration duration separately includes adaptive buffering and
            # dispatch, which can intentionally extend beyond synthesis.
            "generation_total_ms": generation_total_ms,
            "orchestration_total_ms": tts_orchestration_ms,
            "total_ms": generation_total_ms,
            "segments": segment_metrics,
            "dispatches": dispatch_records,
        }

        yield {"type": "tts_done", "meta": tts_meta, "final_audio": final_audio_path}

        timings = {
            "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
            "stt_inference_ms": stt_result.inference_ms if stt_result else None,
            "rag_ms": rag_result.client_latency_ms,
            "rag_internal_ms": rag_result.latency.total_ms,
            "llm_ms": (
                llm_meta.get("client_total_ms")
                or (llm_meta.get("metrics") or {}).get("total_latency_ms")
            ),
            "llm_ttft_ms": (
                llm_meta.get("client_ttft_ms")
                or (llm_meta.get("metrics") or {}).get("ttft_ms")
            ),
            "tts_ms": tts_meta["total_ms"],
            "first_tts_body_ms": first_tts_body_ms,
            "first_audio_ready_ms": first_audio_ready_ms,
            "first_audio_ms": jitter.metrics.first_dispatch_ms,
            "total_ms": (time.perf_counter() - started_total) * 1000.0,
        }

        yield {
            "type": "done",
            "transcript": transcript,
            "answer": "".join(answer_parts).strip(),
            "sources": source_views,
            "timings": timings,
            "details": {
                **base_details,
                "llm": llm_meta,
                "tts": tts_meta,
            },
            "final_audio": final_audio_path,
            "rag_mode": rag_result.mode,
            "rerank_applied": rag_result.rerank_applied,
        }

    def _stream_process_v7(
        self,
        *,
        audio_path: str | Path | None = None,
        query_text: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        synthesize: bool = True,
        rag_mode: str | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
        voice_index: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        started_total = time.perf_counter()
        (
            stt_result,
            transcript,
            rag_result,
            messages,
        ) = self._prepare(
            audio_path=audio_path,
            query_text=query_text,
            history=history,
            rag_mode=rag_mode,
            rerank=rerank,
            top_k=top_k,
        )

        source_views = [
            source.model_dump(mode="json")
            for source in self._sources(rag_result)
        ]
        base_details = {
            "stt": (
                stt_result.model_dump(mode="json")
                if stt_result
                else {"skipped": True}
            ),
            "rag": {
                "mode": rag_result.mode,
                "rerank_requested": rag_result.rerank_requested,
                "rerank_applied": rag_result.rerank_applied,
                "context_count": len(rag_result.contexts),
                "latency": rag_result.latency.model_dump(mode="json"),
            },
        }

        yield {
            "type": "prepared",
            "transcript": transcript,
            "sources": source_views,
            "timings": {
                "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
                "stt_inference_ms": stt_result.inference_ms if stt_result else None,
                "rag_ms": rag_result.client_latency_ms,
                "rag_internal_ms": rag_result.latency.total_ms,
            },
            "details": base_details,
        }

        if not synthesize:
            # Preserve the normal streaming text contract without creating TTS
            # worker state. Reuse the V6-compatible client stream directly.
            answer_parts: list[str] = []
            llm_meta: dict[str, Any] = {}
            for event in self.llm.chat_stream(
                messages,
                temperature=self.settings.llm_temperature,
                top_p=self.settings.llm_top_p,
                max_tokens=self.settings.llm_max_tokens,
            ):
                if event["type"] == "delta":
                    delta = str(event.get("content") or "")
                    answer_parts.append(delta)
                    yield {
                        "type": "llm_delta",
                        "delta": delta,
                        "answer": "".join(answer_parts),
                        "meta": event,
                    }
                elif event["type"] == "done":
                    llm_meta = event
                    yield {
                        "type": "llm_done",
                        "answer": "".join(answer_parts).strip(),
                        "meta": event,
                    }
            timings = {
                "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
                "stt_inference_ms": stt_result.inference_ms if stt_result else None,
                "rag_ms": rag_result.client_latency_ms,
                "rag_internal_ms": rag_result.latency.total_ms,
                "llm_ms": (
                    llm_meta.get("client_total_ms")
                    or (llm_meta.get("metrics") or {}).get("total_latency_ms")
                ),
                "llm_ttft_ms": (
                    llm_meta.get("client_ttft_ms")
                    or (llm_meta.get("metrics") or {}).get("ttft_ms")
                ),
                "tts_ms": 0.0,
                "first_tts_body_ms": None,
                "first_audio_ready_ms": None,
                "first_audio_ms": None,
                "total_ms": (time.perf_counter() - started_total) * 1000.0,
            }
            yield {
                "type": "done",
                "transcript": transcript,
                "answer": "".join(answer_parts).strip(),
                "sources": source_views,
                "timings": timings,
                "details": {
                    **base_details,
                    "llm": llm_meta,
                    "tts": {"skipped": True},
                },
                "final_audio": None,
                "rag_mode": rag_result.mode,
                "rerank_applied": rag_result.rerank_applied,
            }
            return

        control_events: queue.Queue[dict[str, Any]] = queue.Queue()
        pcm_queue: queue.Queue[Any] = queue.Queue(
            maxsize=self.settings.tts_audio_queue_max
        )
        cancel_event = threading.Event()
        llm_done = threading.Event()
        tts_done = threading.Event()
        answer_parts: list[str] = []
        sentence_texts: dict[int, str] = {}
        shared: dict[str, float | None] = {"llm_done_ms": None}

        sentences = SentenceScheduler(
            maxsize=self.settings.tts_sentence_queue_max,
            started_at=started_total,
            cancel_event=cancel_event,
        )
        voice = (
            self.settings.tts_voice_index
            if voice_index is None
            else int(voice_index)
        )
        tts_streamer = TTSStreamWorker(
            tts_client=self.tts,
            voice_index=voice,
            audio_queue=pcm_queue,
            started_at=started_total,
        )
        jitter = AdaptiveStreamJitterController(
            started_at=started_total,
            base_startup_ms=self.settings.tts_startup_buffer_ms,
            min_startup_ms=self.settings.tts_startup_buffer_min_ms,
            max_startup_ms=self.settings.tts_startup_buffer_max_ms,
            startup_max_wait_ms=self.settings.tts_startup_max_wait_ms,
            max_buffer_ahead_ms=self.settings.tts_max_buffer_ahead_ms,
        )
        pcm_buffer = ContinuousPCMBuffer(
            sample_rate=self.settings.tts_sample_rate,
            channels=self.settings.tts_channels,
            dtype=self.settings.tts_dtype,
            target_chunk_ms=self.settings.tts_playback_chunk_ms,
        )

        def llm_worker() -> None:
            chunker = SentenceChunker(
                min_chars=self.settings.tts_phrase_min_chars,
                soft_chars=self.settings.tts_phrase_soft_chars,
                hard_chars=self.settings.tts_phrase_hard_chars,
            )
            first_client_ttft: float | None = None
            try:
                for event in self.llm.chat_stream(
                    messages,
                    temperature=self.settings.llm_temperature,
                    top_p=self.settings.llm_top_p,
                    max_tokens=self.settings.llm_max_tokens,
                ):
                    if event["type"] == "delta":
                        delta = str(event.get("content") or "")
                        if first_client_ttft is None:
                            first_client_ttft = event.get("client_ttft_ms")
                        answer_parts.append(delta)
                        control_events.put(
                            {
                                "type": "llm_delta",
                                "delta": delta,
                                "answer": "".join(answer_parts),
                                "meta": event,
                            }
                        )
                        for sentence in chunker.push(delta):
                            task = sentences.put(sentence)
                            if task is not None:
                                sentence_texts[task.index] = task.text
                    elif event["type"] == "done":
                        for sentence in chunker.flush():
                            task = sentences.put(sentence)
                            if task is not None:
                                sentence_texts[task.index] = task.text
                        shared["llm_done_ms"] = (
                            time.perf_counter() - started_total
                        ) * 1000.0
                        event["client_ttft_ms"] = first_client_ttft
                        event["tts_sentence_count"] = sentences.sentence_count
                        event["tts_phrase_count"] = sentences.sentence_count
                        event["pipeline_done_ms"] = shared["llm_done_ms"]
                        control_events.put(
                            {
                                "type": "llm_done",
                                "answer": "".join(answer_parts).strip(),
                                "meta": event,
                            }
                        )
            except Exception as exc:
                control_events.put(
                    {"type": "error", "stage": "llm", "error": str(exc)}
                )
                cancel_event.set()
            finally:
                llm_done.set()
                sentences.close()

        def tts_worker() -> None:
            tts_streamer.run(sentences)
            if tts_streamer.error is not None:
                cancel_event.set()
            tts_done.set()

        threading.Thread(
            target=llm_worker,
            name="llm-stream-producer-v7",
            daemon=True,
        ).start()
        threading.Thread(
            target=tts_worker,
            name="tts-native-stream-worker-v7",
            daemon=True,
        ).start()

        llm_meta: dict[str, Any] = {}
        had_error = False
        audio_source_done = False
        final_pcm_parts: list[bytes] = []
        dispatch_records: list[dict[str, Any]] = []
        first_pcm_ms: float | None = None
        latest_rolling_rtf: float | None = None
        streaming_modes: set[str] = set()
        upstream_backends: set[str] = set()

        def make_audio_event(dispatch: PCMDispatch) -> dict[str, Any]:
            estimated_gap_ms, buffer_ahead_ms = jitter.record_dispatch(
                dispatch.duration_ms
            )
            dispatch_ms = (time.perf_counter() - started_total) * 1000.0
            sentence_indices = list(dispatch.sentence_indices)
            dispatch_records.append(
                {
                    "dispatch_ms": dispatch_ms,
                    "duration_ms": dispatch.duration_ms,
                    "sentence_indices": sentence_indices,
                }
            )
            return {
                "type": "audio",
                "audio": pcm_to_wav_bytes(
                    dispatch.raw,
                    dtype=self.settings.tts_dtype,
                    channels=self.settings.tts_channels,
                    sample_rate=self.settings.tts_sample_rate,
                ),
                "sentence": " ".join(
                    sentence_texts.get(i, "") for i in sentence_indices
                ).strip(),
                "sentences": [
                    sentence_texts.get(i, "") for i in sentence_indices
                ],
                "sentence_indices": sentence_indices,
                "audio_chunk": len(dispatch_records),
                "duration_ms": dispatch.duration_ms,
                "generation_mode": "stream",
                "streaming_mode": (
                    "frame-pcm" if "frame-pcm" in streaming_modes else "stream"
                ),
                "upstream_backend": (
                    sorted(upstream_backends)[0]
                    if upstream_backends
                    else "unknown"
                ),
                "playout_strategy": "continuous-pcm-adaptive-jitter-v7",
                "estimated_server_gap_ms": estimated_gap_ms,
                "estimated_buffer_ahead_ms": buffer_ahead_ms,
                "startup_buffer_target_ms": jitter.metrics.startup_target_ms,
                "sentence_queue_depth": sentences.qsize(),
                "audio_queue_depth": pcm_queue.qsize(),
            }

        while True:
            progressed = False

            for _ in range(32):
                try:
                    event = control_events.get_nowait()
                except queue.Empty:
                    break
                progressed = True
                if event["type"] == "llm_done":
                    llm_meta = event.get("meta") or {}
                elif event["type"] == "error":
                    had_error = True
                    cancel_event.set()
                yield event

            can_pull_audio = (
                jitter.metrics.first_dispatch_ms is None
                or (
                    pcm_buffer.buffered_ms + jitter.buffer_ahead_ms()
                    < self.settings.tts_max_buffer_ahead_ms
                )
                or tts_done.is_set()
            )

            if not audio_source_done and can_pull_audio:
                try:
                    audio_item = pcm_queue.get(
                        timeout=0.01 if not progressed else 0.0
                    )
                    progressed = True
                except queue.Empty:
                    audio_item = "__empty__"

                if audio_item is None:
                    audio_source_done = True
                elif isinstance(audio_item, dict) and audio_item.get("type") == "error":
                    had_error = True
                    cancel_event.set()
                    yield audio_item
                elif audio_item != "__empty__":
                    assert isinstance(audio_item, PCMStreamChunk)
                    pcm_buffer.append(
                        audio_item.raw,
                        sentence_index=audio_item.sentence_index,
                    )
                    final_pcm_parts.append(audio_item.raw)
                    latest_rolling_rtf = audio_item.rolling_rtf
                    streaming_modes.add(audio_item.streaming_mode)
                    upstream_backends.add(audio_item.upstream_backend)
                    if first_pcm_ms is None:
                        first_pcm_ms = audio_item.first_pcm_ms

            if (
                pcm_buffer.buffered_bytes
                and jitter.metrics.first_dispatch_ms is None
            ):
                if jitter.startup_ready(
                    buffered_ms=pcm_buffer.buffered_ms,
                    rolling_rtf=latest_rolling_rtf,
                    source_done=audio_source_done,
                ):
                    # First V7 dispatch may be smaller than the normal browser
                    # chunk target. This is intentional: TTFA is bounded by the
                    # adaptive startup target/max-wait instead of full-sentence
                    # synthesis or a fixed 1+ second media chunk.
                    dispatch = pcm_buffer.pop(force=True)
                    if dispatch is not None:
                        yield make_audio_event(dispatch)
                        progressed = True

            elif (
                pcm_buffer.buffered_bytes
                and jitter.metrics.first_dispatch_ms is not None
            ):
                ahead_ms = jitter.buffer_ahead_ms()
                if (
                    pcm_buffer.ready
                    and ahead_ms <= self.settings.tts_max_buffer_ahead_ms
                ):
                    dispatch = pcm_buffer.pop(force=False)
                    if dispatch is not None:
                        yield make_audio_event(dispatch)
                        progressed = True
                elif (
                    ahead_ms <= self.settings.tts_playback_min_chunk_ms / 2
                    and pcm_buffer.ready_for(self.settings.tts_playback_min_chunk_ms)
                ):
                    # V7 rescue dispatch: when the estimated browser cushion is
                    # nearly empty, do not wait for a full normal media chunk.
                    # A bounded smaller chunk is preferable to a server-side gap.
                    dispatch = pcm_buffer.pop(
                        force=False,
                        target_ms=self.settings.tts_playback_min_chunk_ms,
                    )
                    if dispatch is not None:
                        yield make_audio_event(dispatch)
                        progressed = True
                elif audio_source_done:
                    dispatch = pcm_buffer.pop(force=True)
                    if dispatch is not None:
                        yield make_audio_event(dispatch)
                        progressed = True

            finished = (
                llm_done.is_set()
                and tts_done.is_set()
                and audio_source_done
                and pcm_buffer.buffered_bytes == 0
                and control_events.empty()
            )
            if finished:
                break

            if not progressed:
                time.sleep(0.005)

        if had_error:
            return

        final_audio_path: str | None = None
        if final_pcm_parts:
            raw_final = b"".join(final_pcm_parts)
            save_fn = getattr(self.tts, "save_pcm_wav", None)
            if callable(save_fn):
                final_audio_path = save_fn(raw_final, prefix="v7-final")
            else:
                import uuid

                output_dir = Path(self.settings.tts_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                final_path = output_dir / f"tts-v7-final-{uuid.uuid4().hex}.wav"
                final_path.write_bytes(
                    pcm_to_wav_bytes(
                        raw_final,
                        dtype=self.settings.tts_dtype,
                        channels=self.settings.tts_channels,
                        sample_rate=self.settings.tts_sample_rate,
                    )
                )
                final_audio_path = str(final_path)

        results = list(tts_streamer.results)
        rtfs = [item.rtf for item in results if item.rtf is not None]
        total_audio_ms = sum(item.audio_duration_ms for item in results)
        generation_total_ms = (
            max(item.synthesis_finished_ms for item in results)
            - min(item.synthesis_started_ms for item in results)
            if results
            else 0.0
        )

        first_dispatch_ms = jitter.metrics.first_dispatch_ms
        cross_sentence_dispatch_count = sum(
            1 for record in dispatch_records
            if len(record.get("sentence_indices") or []) > 1
        )
        stream_overlap_count = 0
        next_first_pcm_before_previous_end_count = 0
        cumulative_audio_ms = 0.0
        if first_dispatch_ms is not None:
            for index, item in enumerate(results):
                if index > 0:
                    estimated_previous_end = (
                        first_dispatch_ms + cumulative_audio_ms
                    )
                    if item.synthesis_started_ms < estimated_previous_end:
                        stream_overlap_count += 1
                    if (
                        item.first_pcm_ms is not None
                        and item.first_pcm_ms < estimated_previous_end
                    ):
                        next_first_pcm_before_previous_end_count += 1
                cumulative_audio_ms += item.audio_duration_ms

        llm_done_ms = shared.get("llm_done_ms")
        first_tts_request_ms = (
            results[0].synthesis_started_ms if results else None
        )
        first_tts_pcm_ms = (
            results[0].first_pcm_ms if results else first_pcm_ms
        )

        segments = [
            {
                "sentence": item.index,
                "chars": len(item.sentence),
                "sentence_text": item.sentence,
                "queued_ms": item.queued_ms,
                "sentence_wait_ms": item.sentence_wait_ms,
                "synthesis_started_ms": item.synthesis_started_ms,
                "first_pcm_ms": item.first_pcm_ms,
                "synthesis_finished_ms": item.synthesis_finished_ms,
                "synthesis_ms": item.synthesis_ms,
                "audio_duration_ms": item.audio_duration_ms,
                "ttfb_ms": item.ttfb_ms,
                "rtf": item.rtf,
                "chunk_count": item.chunk_count,
                "generation_mode": item.generation_mode,
                "streaming_mode": item.streaming_mode,
                "upstream_backend": item.upstream_backend,
                "sentence_queue_depth_at_enqueue": (
                    item.sentence_queue_depth_at_enqueue
                ),
                "sentence_queue_depth_at_start": (
                    item.sentence_queue_depth_at_start
                ),
            }
            for item in results
        ]

        tts_meta = {
            "pipeline_version": "v7",
            "segmentation_mode": "sentence-first",
            "generation_strategy": "sentence-native-stream-v7",
            "generation_mode": "stream",
            "generation_modes": sorted(
                {item.generation_mode for item in results}
            ),
            "streaming_modes": sorted(
                {item.streaming_mode for item in results}
            ),
            "upstream_backends": sorted(
                {item.upstream_backend for item in results}
            ),
            "native_frame_streaming": any(
                item.streaming_mode == "frame-pcm" for item in results
            ),
            "playout_strategy": "continuous-pcm-adaptive-jitter-v7",
            "sentence_count": len(results),
            "segment_count": len(results),
            "stream_chunk_count": tts_streamer.total_stream_chunks,
            "playback_chunk_count": len(dispatch_records),
            "cross_sentence_dispatch_count": cross_sentence_dispatch_count,
            "playback_audio_ms": total_audio_ms,
            "sentence_queue_capacity": self.settings.tts_sentence_queue_max,
            "max_sentence_queue_depth": sentences.max_depth,
            "sentence_backpressure_ms": sentences.total_backpressure_ms,
            "audio_queue_capacity": self.settings.tts_audio_queue_max,
            "max_audio_queue_depth": tts_streamer.max_audio_queue_depth,
            "audio_queue_backpressure_ms": (
                tts_streamer.total_audio_queue_backpressure_ms
            ),
            "startup_buffer_target_ms": jitter.metrics.startup_target_ms,
            "startup_buffer_observed_ms": jitter.metrics.startup_buffer_ms,
            "max_buffer_ahead_ms": jitter.metrics.max_buffer_ahead_ms,
            "estimated_underrun_count": jitter.metrics.estimated_underrun_count,
            "estimated_max_gap_ms": jitter.metrics.estimated_max_gap_ms,
            "sentence_stream_overlap_count": stream_overlap_count,
            "sentence_stream_overlap_ratio": (
                stream_overlap_count / max(1, len(results) - 1)
                if len(results) > 1
                else 0.0
            ),
            "next_sentence_first_pcm_before_previous_end_count": (
                next_first_pcm_before_previous_end_count
            ),
            "next_sentence_first_pcm_before_previous_end_ratio": (
                next_first_pcm_before_previous_end_count
                / max(1, len(results) - 1)
                if len(results) > 1
                else 0.0
            ),
            # Compatibility aliases used by the existing V6 metrics UI.
            "prefetch_overlap_count": stream_overlap_count,
            "prefetch_overlap_ratio": (
                stream_overlap_count / max(1, len(results) - 1)
                if len(results) > 1
                else 0.0
            ),
            "prefetch_ready_before_previous_end_count": (
                next_first_pcm_before_previous_end_count
            ),
            "prefetch_ready_before_previous_end_ratio": (
                next_first_pcm_before_previous_end_count
                / max(1, len(results) - 1)
                if len(results) > 1
                else 0.0
            ),
            "llm_done_ms": llm_done_ms,
            "first_sentence_queued_ms": (
                results[0].queued_ms if results else None
            ),
            "pipeline_first_tts_request_ms": first_tts_request_ms,
            "pipeline_first_tts_pcm_ms": first_tts_pcm_ms,
            "pipeline_first_tts_body_ms": first_tts_pcm_ms,
            "pipeline_first_audio_ready_ms": first_tts_pcm_ms,
            "pipeline_first_audio_ms": first_dispatch_ms,
            "first_pcm_to_audio_dispatch_ms": (
                first_dispatch_ms - first_tts_pcm_ms
                if first_dispatch_ms is not None and first_tts_pcm_ms is not None
                else None
            ),
            "first_audio_before_first_sentence_finished": (
                bool(
                    first_dispatch_ms is not None
                    and results
                    and first_dispatch_ms < results[0].synthesis_finished_ms
                )
                if results else None
            ),
            "first_tts_pcm_before_llm_done": (
                bool(
                    first_tts_pcm_ms is not None
                    and llm_done_ms is not None
                    and first_tts_pcm_ms < llm_done_ms
                )
                if llm_done_ms is not None
                else None
            ),
            "first_audio_before_llm_done": (
                bool(
                    first_dispatch_ms is not None
                    and llm_done_ms is not None
                    and first_dispatch_ms < llm_done_ms
                )
                if llm_done_ms is not None
                else None
            ),
            "mean_rtf": (sum(rtfs) / len(rtfs)) if rtfs else None,
            "max_rtf": max(rtfs) if rtfs else None,
            "trimmed_silence_ms": 0.0,
            "streaming_edge_trim_applied": False,
            "final_audio_path": final_audio_path,
            "generation_total_ms": generation_total_ms,
            "orchestration_total_ms": (
                time.perf_counter() - started_total
            ) * 1000.0,
            "total_ms": generation_total_ms,
            "segments": segments,
            "dispatches": dispatch_records,
        }

        yield {
            "type": "tts_done",
            "meta": tts_meta,
            "final_audio": final_audio_path,
        }

        timings = {
            "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
            "stt_inference_ms": stt_result.inference_ms if stt_result else None,
            "rag_ms": rag_result.client_latency_ms,
            "rag_internal_ms": rag_result.latency.total_ms,
            "llm_ms": (
                llm_meta.get("client_total_ms")
                or (llm_meta.get("metrics") or {}).get("total_latency_ms")
            ),
            "llm_ttft_ms": (
                llm_meta.get("client_ttft_ms")
                or (llm_meta.get("metrics") or {}).get("ttft_ms")
            ),
            "tts_ms": tts_meta["total_ms"],
            "first_tts_body_ms": first_tts_pcm_ms,
            "first_audio_ready_ms": first_tts_pcm_ms,
            "first_audio_ms": first_dispatch_ms,
            "total_ms": (time.perf_counter() - started_total) * 1000.0,
        }

        yield {
            "type": "done",
            "transcript": transcript,
            "answer": "".join(answer_parts).strip(),
            "sources": source_views,
            "timings": timings,
            "details": {
                **base_details,
                "llm": llm_meta,
                "tts": tts_meta,
            },
            "final_audio": final_audio_path,
            "rag_mode": rag_result.mode,
            "rerank_applied": rag_result.rerank_applied,
        }

    def stream_process(
        self,
        *,
        audio_path: str | Path | None = None,
        query_text: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        synthesize: bool = True,
        rag_mode: str | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
        voice_index: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Dispatch realtime turns to V7 by default, retaining V6 compatibility.

        ``TTS_GENERATION_MODE=stream`` activates V7 sentence-native PCM
        streaming. ``sentence``/``auto`` retain the V6 complete-sentence
        buffering path for A/B testing and compatibility.
        """
        kwargs = {
            "audio_path": audio_path,
            "query_text": query_text,
            "history": history,
            "synthesize": synthesize,
            "rag_mode": rag_mode,
            "rerank": rerank,
            "top_k": top_k,
            "voice_index": voice_index,
        }
        if synthesize and self.settings.tts_generation_mode == "stream":
            yield from self._stream_process_v7(**kwargs)
            return
        yield from self._stream_process_v6(**kwargs)

