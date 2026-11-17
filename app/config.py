"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

class Settings(BaseModel):
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        pattern=r"^https?://",
    )
    ollama_model: str = Field(default="qwen3.5:4b", min_length=1)
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        min_length=1,
    )
    openai_api_key: str | None = None
    openai_model: str | None = None


def load_settings(env_file: str = ".env") -> Settings:
    load_dotenv(env_file)

    return Settings(
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        ollama_embedding_model=os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text",
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL"),
    )