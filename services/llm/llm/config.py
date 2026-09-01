from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_name: str = "qwen3.5:4b"

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_timeout: float = 180.0

    api_host: str = "0.0.0.0"
    api_port: int = 8002

    default_temperature: float = 0.3
    default_top_p: float = 0.9
    default_max_tokens: int = 300

    log_level: str = "INFO"
    log_request_id: bool = True
    log_prompts: bool = False

    enable_metrics: bool = True
