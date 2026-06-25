from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    INDEX_DIR: Path = BASE_DIR / "index_store"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TABLE_ROW_BATCH_SIZE: int = 5

    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # LLM Backend — "groq" (default) or "ollama" (fallback)
    LLM_BACKEND: Literal["groq", "ollama"] = "groq"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS: int = 512
    GROQ_TEMPERATURE: float = 0.1

    # Ollama (fallback)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi4-mini"
    OLLAMA_MODEL_7B: str = "phi4-mini"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_MAX_TOKENS: int = 512
    OLLAMA_TEMPERATURE: float = 0.1

    # Retrieval
    RETRIEVER_TOP_K: int = 15
    RERANKER_TOP_K: int = 5

    # API Security
    API_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()