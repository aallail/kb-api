"""Application configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Anthropic API (for LLM only)
    ANTHROPIC_API_KEY: str = ""

    # Database
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@db:5432/kb"

    # API Security
    API_KEY: str = "dev-key"

    # Document processing
    MAX_UPLOAD_SIZE_MB: int = 10
    DEFAULT_TOP_K: int = 6
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MIN_SIMILARITY_SCORE: float = 0.3  # Minimum cosine similarity score (0-1)

    # Embedding model (LOCAL - sentence-transformers)
    # Using all-MiniLM-L6-v2 (384d) - fast and good quality
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384 dims
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension

    # LLM model (API - Claude)
    LLM_MODEL: str = "claude-3-haiku-20240307"  # Cheapest Claude model
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )


settings = Settings()
