from __future__ import annotations


class VoiceAppError(RuntimeError):
    """Base error for integration failures."""


class ServiceError(VoiceAppError):
    def __init__(
        self,
        service: str,
        message: str,
        *,
        status_code: int | None = None,
        details: object | None = None,
    ) -> None:
        self.service = service
        self.status_code = status_code
        self.details = details
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"{service}: {message}{suffix}")


class ContractError(VoiceAppError):
    """Raised when an upstream API response violates the verified contract."""


class ConfigurationError(VoiceAppError):
    """Raised for invalid integration configuration."""
