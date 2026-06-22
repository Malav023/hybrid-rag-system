from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    INDEX_DIR: Path = BASE_DIR / "index_store"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TABLE_ROW_BATCH_SIZE: int = 5  # rows per chunk for large tables

    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder:1.5b"

    class Config:
        env_file = ".env"

settings = Settings()