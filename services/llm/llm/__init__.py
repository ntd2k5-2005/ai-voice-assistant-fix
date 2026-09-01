"""
LLM Service Package
Handles Ollama model interactions and metrics tracking
"""

from llm.service import LLMService
from llm.config import Settings
from llm.schemas import (
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    Message,
)
from llm.errors import (
    LLMError,
    OllamaUnavailableError,
    ModelNotFoundError,
    GenerationTimeoutError,
)

__all__ = [
    "LLMService",
    "Settings",
    "GenerateRequest",
    "GenerateResponse",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "LLMError",
    "OllamaUnavailableError",
    "ModelNotFoundError",
    "GenerationTimeoutError",
]
