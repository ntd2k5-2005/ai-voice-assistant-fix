from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .contracts import ChatMessage


class TurnState(StrEnum):
    CREATED = "created"
    TRANSCRIBING = "transcribing"
    RETRIEVING = "retrieving"
    THINKING = "thinking"
    SPEAKING = "speaking"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_STATES = {
    TurnState.COMPLETED,
    TurnState.FAILED,
    TurnState.CANCELLED,
    TurnState.INTERRUPTED,
}

_TRANSITIONS: dict[TurnState, set[TurnState]] = {
    TurnState.CREATED: {
        TurnState.TRANSCRIBING,
        TurnState.RETRIEVING,
        TurnState.CANCELLED,
        TurnState.INTERRUPTED,
        TurnState.FAILED,
    },
    TurnState.TRANSCRIBING: {
        TurnState.RETRIEVING,
        TurnState.CANCELLED,
        TurnState.INTERRUPTED,
        TurnState.FAILED,
    },
    TurnState.RETRIEVING: {
        TurnState.THINKING,
        TurnState.CANCELLED,
        TurnState.INTERRUPTED,
        TurnState.FAILED,
    },
    TurnState.THINKING: {
        TurnState.SPEAKING,
        TurnState.COMPLETED,
        TurnState.CANCELLED,
        TurnState.INTERRUPTED,
        TurnState.FAILED,
    },
    TurnState.SPEAKING: {
        TurnState.COMPLETED,
        TurnState.CANCELLED,
        TurnState.INTERRUPTED,
        TurnState.FAILED,
    },
}


@dataclass
class TurnRecord:
    turn_id: str
    generation: int
    state: TurnState = TurnState.CREATED
    user_text: str = ""
    assistant_text: str = ""
    sources: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "generation": self.generation,
            "state": self.state.value,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "sources": list(self.sources),
            "grounding": dict(self.grounding),
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ConversationRecord:
    conversation_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    generation: int = 0
    turns: list[TurnRecord] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "generation": self.generation,
            "turns": [turn.as_dict() for turn in self.turns],
        }


class ConversationHistory(Protocol):
    """Persistence boundary; SQLite/Redis can replace this implementation."""

    def create(self) -> ConversationRecord: ...
    def get(self, conversation_id: str) -> ConversationRecord: ...
    def delete(self, conversation_id: str) -> bool: ...
    def begin_turn(self, conversation_id: str, turn_id: str) -> TurnRecord: ...
    def transition(self, conversation_id: str, turn_id: str, state: TurnState, **updates) -> TurnRecord: ...
    def messages(self, conversation_id: str, max_turns: int) -> Sequence[ChatMessage]: ...
    def prune(self) -> int: ...


class InMemoryConversationHistory:
    def __init__(self, *, max_conversations: int = 64, ttl_s: float = 6 * 3600) -> None:
        self.max_conversations = max(1, max_conversations)
        self.ttl_s = max(60.0, ttl_s)
        self._items: dict[str, ConversationRecord] = {}
        self._lock = threading.RLock()

    def create(self) -> ConversationRecord:
        with self._lock:
            self._prune_locked()
            conversation = ConversationRecord(conversation_id=uuid.uuid4().hex)
            self._items[conversation.conversation_id] = conversation
            return conversation

    def get(self, conversation_id: str) -> ConversationRecord:
        with self._lock:
            item = self._items.get(conversation_id)
            if item is None:
                raise KeyError(conversation_id)
            return item

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._items.pop(conversation_id, None) is not None

    def begin_turn(self, conversation_id: str, turn_id: str) -> TurnRecord:
        with self._lock:
            conversation = self.get(conversation_id)
            active = [turn for turn in conversation.turns if turn.state not in TERMINAL_STATES]
            if active:
                raise RuntimeError(f"conversation already has active turn {active[-1].turn_id}")
            conversation.generation += 1
            conversation.updated_at = time.time()
            turn = TurnRecord(turn_id=turn_id, generation=conversation.generation)
            conversation.turns.append(turn)
            return turn

    def transition(
        self,
        conversation_id: str,
        turn_id: str,
        state: TurnState,
        **updates,
    ) -> TurnRecord:
        with self._lock:
            conversation = self.get(conversation_id)
            turn = next((item for item in conversation.turns if item.turn_id == turn_id), None)
            if turn is None:
                raise KeyError(turn_id)
            if turn.state != state:
                allowed = _TRANSITIONS.get(turn.state, set())
                if state not in allowed:
                    if turn.state in TERMINAL_STATES:
                        return turn
                    raise RuntimeError(f"invalid turn transition {turn.state.value} -> {state.value}")
                turn.state = state
            for key in ("user_text", "assistant_text", "sources", "grounding", "error"):
                if key in updates:
                    setattr(turn, key, updates[key])
            now = time.time()
            turn.updated_at = now
            conversation.updated_at = now
            return turn

    def messages(self, conversation_id: str, max_turns: int) -> Sequence[ChatMessage]:
        with self._lock:
            conversation = self.get(conversation_id)
            complete = [turn for turn in conversation.turns if turn.state == TurnState.COMPLETED]
            rows: list[ChatMessage] = []
            for turn in complete[-max(0, max_turns) :]:
                if turn.user_text and turn.assistant_text:
                    rows.append(ChatMessage(role="user", content=turn.user_text))
                    rows.append(ChatMessage(role="assistant", content=turn.assistant_text))
            return rows

    def prune(self) -> int:
        with self._lock:
            return self._prune_locked()

    def _prune_locked(self) -> int:
        now = time.time()
        removable = [
            item for item in self._items.values()
            if now - item.updated_at >= self.ttl_s
            and all(turn.state in TERMINAL_STATES for turn in item.turns)
        ]
        if len(self._items) - len(removable) > self.max_conversations:
            survivors = sorted(
                (item for item in self._items.values() if item not in removable),
                key=lambda item: item.updated_at,
            )
            overflow = len(self._items) - len(removable) - self.max_conversations
            removable.extend(
                item for item in survivors[:overflow]
                if all(turn.state in TERMINAL_STATES for turn in item.turns)
            )
        for item in removable:
            self._items.pop(item.conversation_id, None)
        return len(removable)
