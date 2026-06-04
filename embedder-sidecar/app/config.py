from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#https://huggingface.co/google/embeddinggemma-300m

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    embed_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    embed_dim: int = Field(default=1024, ge=64, le=8192)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)
    log_level: str = Field(default="INFO")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
