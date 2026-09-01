from voice_app.config import DEFAULT_SYSTEM_PROMPT
from voice_app.contracts import RAGContext
from voice_app.prompt_builder import PromptBuilder


def test_prompt_contains_question_and_sources():
    builder = PromptBuilder(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_context_chars_per_item=1000,
    )
    contexts = [
        RAGContext(
            rank=1,
            chunk_id="c1",
            document_id="d1",
            text="Nội dung pháp lý cần thiết.",
            source="doc.pdf",
            page=3,
            section="Điều 1",
            score=0.91,
            score_source="dense",
        )
    ]
    messages = builder.build(
        question="Quy định là gì?",
        contexts=contexts,
    )
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert "Quy định là gì?" in messages[-1].content
    assert "[Nguồn 1]" in messages[-1].content
    assert "Nội dung pháp lý cần thiết." in messages[-1].content
