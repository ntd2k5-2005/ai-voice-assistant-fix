from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from .contracts import RAGContext


NO_ANSWER_VI = "Tôi chưa tìm thấy thông tin này trong kho tri thức hiện tại."


class QueryIntent(StrEnum):
    SOCIAL = "social"
    FACTUAL = "factual"


def _plain(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", value).strip()


_SOCIAL_PATTERNS = (
    re.compile(r"^(xin\s+chao|chao|hello|hi)(\s+(ban|eureka|tro\s+ly))?$"),
    re.compile(r"^(cam\s+on|thank\s+you|thanks)(\s+(ban|eureka))?$"),
    re.compile(r"^(tam\s+biet|bye|goodbye)$"),
    re.compile(r"^(ban\s+la\s+ai|ban\s+co\s+the\s+lam\s+gi)$"),
)


@dataclass(frozen=True)
class GroundingDecision:
    intent: QueryIntent
    context_sufficient: bool
    policy: str
    reason: str
    forced_answer: str | None = None
    calibration_status: str = "NOT_EVALUATED"

    def as_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "context_sufficient": self.context_sufficient,
            "policy": self.policy,
            "reason": self.reason,
            "forced_answer": self.forced_answer,
            "calibration_status": self.calibration_status,
        }


class GroundingPolicy:
    """Conservative answerability gate without an invented score threshold.

    Public retrieval scores are normalized per retrieval mode and no labelled
    answerability corpus exists in this repository. Treating any fixed score as
    calibrated would therefore be misleading. This gate makes only decisions
    supported by deterministic evidence and reports calibration honestly.
    """

    def intent(self, query: str) -> QueryIntent:
        normalized = _plain(query)
        if any(pattern.fullmatch(normalized) for pattern in _SOCIAL_PATTERNS):
            return QueryIntent.SOCIAL
        return QueryIntent.FACTUAL

    def decide(
        self,
        query: str,
        contexts: Sequence[RAGContext],
    ) -> GroundingDecision:
        intent = self.intent(query)
        if intent == QueryIntent.SOCIAL:
            return GroundingDecision(
                intent=intent,
                context_sufficient=True,
                policy="social-bypass",
                reason="Greeting/social interaction does not require document evidence.",
            )
        usable = [context for context in contexts if context.text.strip()]
        if not usable:
            return GroundingDecision(
                intent=intent,
                context_sufficient=False,
                policy="deterministic-no-answer",
                reason="Factual query has no usable retrieved context.",
                forced_answer=NO_ANSWER_VI,
            )
        return GroundingDecision(
            intent=intent,
            context_sufficient=True,
            policy="context-present-prompt-grounding",
            reason=(
                "Usable contexts and provenance are present; relevance score "
                "threshold remains uncalibrated until labelled data is supplied."
            ),
        )
