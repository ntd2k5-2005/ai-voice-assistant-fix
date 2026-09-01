"""
Custom exceptions for LLM service
Provides clear error handling for different failure scenarios
"""


class LLMError(Exception):
    """Base exception for all LLM service errors"""
    pass


class OllamaUnavailableError(LLMError):
    """Raised when Ollama server is not reachable"""
    def __init__(self, url: str, original_error: Exception = None):
        self.url = url
        self.original_error = original_error
        message = f"Ollama server unavailable at {url}"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message)


class ModelNotFoundError(LLMError):
    """Raised when requested model is not available in Ollama"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' not found. Please run: ollama pull {model_name}")


class GenerationTimeoutError(LLMError):
    """Raised when generation exceeds timeout"""
    def __init__(self, timeout: int):
        self.timeout = timeout
        super().__init__(f"Generation exceeded timeout of {timeout} seconds")


class InvalidRequestError(LLMError):
    """Raised when request parameters are invalid"""
    pass
