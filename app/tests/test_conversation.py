from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from voice_app.conversation import InMemoryConversationHistory, TurnState
from voice_app.web import TurnManager


def test_history_is_isolated_and_only_completed_turns_reach_the_prompt():
    store = InMemoryConversationHistory()
    first = store.create()
    second = store.create()
    turn = store.begin_turn(first.conversation_id, "t1")
    assert turn.generation == 1
    store.transition(first.conversation_id, "t1", TurnState.RETRIEVING, user_text="Xin chào")
    store.transition(first.conversation_id, "t1", TurnState.THINKING)
    store.transition(
        first.conversation_id,
        "t1",
        TurnState.COMPLETED,
        assistant_text="Chào bạn",
        sources=[{"document_id": "welcome"}],
    )

    assert [message.content for message in store.messages(first.conversation_id, 3)] == [
        "Xin chào",
        "Chào bạn",
    ]
    assert store.messages(second.conversation_id, 3) == []


def test_conversation_rejects_overlapping_turn_and_invalid_transition():
    store = InMemoryConversationHistory()
    conversation = store.create()
    store.begin_turn(conversation.conversation_id, "active")
    with pytest.raises(RuntimeError, match="active turn"):
        store.begin_turn(conversation.conversation_id, "overlap")
    with pytest.raises(RuntimeError, match="invalid turn transition"):
        store.transition(conversation.conversation_id, "active", TurnState.COMPLETED)


class _Pipeline:
    def __init__(self) -> None:
        self.histories: list[list[str]] = []

    def stream_turn(self, *, query_text, history, cancel_event, **_kwargs):
        self.histories.append([message.content for message in history])
        yield {"type": "stage", "stage": "rag"}
        yield {"type": "transcript_ready", "transcript": query_text}
        yield {"type": "prepared", "transcript": query_text, "sources": [{"document_id": "doc"}]}
        yield {"type": "llm_delta", "answer": f"Trả lời {query_text}"}
        yield {"type": "pcm", "raw": np.zeros(16, dtype="<f4").tobytes(), "chunk_index": 1}
        yield {
            "type": "done",
            "transcript": query_text,
            "answer": f"Trả lời {query_text}",
            "sources": [{"document_id": "doc"}],
        }


class _CancelledPipeline:
    def __init__(self) -> None:
        self.started = threading.Event()

    def stream_turn(self, *, query_text, cancel_event, **_kwargs):
        yield {"type": "stage", "stage": "rag"}
        yield {"type": "transcript_ready", "transcript": query_text}
        yield {"type": "prepared", "transcript": query_text, "sources": []}
        self.started.set()
        assert cancel_event.wait(2)
        # Simulate an uncooperative downstream yielding stale work. The manager
        # is the final ownership gate and must discard both control and PCM.
        yield {"type": "llm_delta", "answer": "stale"}
        yield {"type": "pcm", "raw": np.zeros(16, dtype="<f4").tobytes(), "chunk_index": 1}
        yield {"type": "done", "transcript": query_text, "answer": "stale", "sources": []}


def _manager(pipeline):
    settings = SimpleNamespace(
        max_history_turns=3,
        tts_audio_queue_max=4,
    )
    orchestrator = SimpleNamespace(settings=settings)
    manager = TurnManager(orchestrator)
    manager.pipeline = pipeline
    return manager


def test_manager_supplies_multi_turn_history_and_provenance():
    pipeline = _Pipeline()
    manager = _manager(pipeline)
    first = manager.start(query_text="Một")
    assert first.finished.wait(2)
    second = manager.start(conversation_id=first.conversation_id, query_text="Hai")
    assert second.finished.wait(2)

    assert pipeline.histories == [[], ["Một", "Trả lời Một"]]
    conversation = manager.history.get(first.conversation_id).as_dict()
    assert [turn["state"] for turn in conversation["turns"]] == ["completed", "completed"]
    assert conversation["turns"][0]["sources"] == [{"document_id": "doc"}]


def test_barge_in_cancellation_discards_stale_events_and_audio():
    pipeline = _CancelledPipeline()
    manager = _manager(pipeline)
    session = manager.start(query_text="Lượt cũ")
    assert pipeline.started.wait(2)

    manager.cancel(session.turn_id, reason="barge_in")
    assert session.finished.wait(2)
    turn = manager.history.get(session.conversation_id).turns[0]
    assert turn.state == TurnState.INTERRUPTED
    assert turn.assistant_text == ""

    logged_types = [event["type"] for event in session.event_log]
    assert "interrupted" in logged_types
    assert "llm_delta" not in logged_types
    assert "pcm" not in logged_types
    assert session.audio_queue.get_nowait() is None


def test_control_event_log_can_resume_after_a_cursor():
    pipeline = _Pipeline()
    manager = _manager(pipeline)
    session = manager.start(query_text="reconnect")
    assert session.finished.wait(2)
    first = manager.next_event(session, 0, 0)
    assert first is not None
    second = manager.next_event(session, first["event_sequence"], 0)
    assert second is not None
    assert second["event_sequence"] == first["event_sequence"] + 1
