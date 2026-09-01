from __future__ import annotations

from collections.abc import Sequence

from .contracts import ChatMessage, RAGContext


class PromptBuilder:
    def __init__(
        self,
        *,
        system_prompt: str,
        max_context_chars_per_item: int = 3500,
        max_history_turns: int = 3,
    ) -> None:
        self.system_prompt = system_prompt.strip()
        self.max_context_chars_per_item = max(200, max_context_chars_per_item)
        self.max_history_turns = max(0, max_history_turns)

    def _context_block(self, contexts: Sequence[RAGContext]) -> str:
        if not contexts:
            return "(Không có ngữ cảnh nào được truy xuất.)"

        parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            meta = [
                f"document_id={ctx.document_id}",
                f"score={ctx.score:.4f}",
            ]
            if ctx.source:
                meta.append(f"source={ctx.source}")
            if ctx.section:
                meta.append(f"section={ctx.section}")
            if ctx.page is not None:
                meta.append(f"page={ctx.page}")
            if ctx.heading_path:
                meta.append("heading=" + " > ".join(ctx.heading_path))

            text = ctx.text.strip()
            if len(text) > self.max_context_chars_per_item:
                text = text[: self.max_context_chars_per_item].rstrip() + "…"

            parts.append(
                f"[Nguồn {i}] ({'; '.join(meta)})\n{text}"
            )
        return "\n\n".join(parts)

    def build(
        self,
        *,
        question: str,
        contexts: Sequence[RAGContext],
        history: Sequence[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=self.system_prompt)]

        if history and self.max_history_turns:
            # A turn is roughly user+assistant, so keep at most 2*N messages.
            keep = list(history)[-(self.max_history_turns * 2) :]
            messages.extend(keep)

        user_content = (
            "CÂU HỎI HIỆN TẠI:\n"
            f"{question.strip()}\n\n"
            "NGỮ CẢNH ĐƯỢC TRUY XUẤT:\n"
            f"{self._context_block(contexts)}\n\n"
            "YÊU CẦU:\n"
            "- Trả lời trực tiếp câu hỏi bằng tiếng Việt.\n"
            "- Chỉ dựa trên ngữ cảnh ở trên đối với thông tin cần căn cứ tài liệu.\n"
            "- Nếu ngữ cảnh không đủ, nói rõ là chưa đủ dữ liệu thay vì suy đoán.\n"
            "- Ưu tiên câu trả lời dễ nghe khi được chuyển thành giọng nói."
        )
        messages.append(ChatMessage(role="user", content=user_content))
        return messages
