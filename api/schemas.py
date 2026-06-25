from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
#  Ingest                                                                      #
# --------------------------------------------------------------------------- #

class IngestResponse(BaseModel):
    source_file: str
    chunks_indexed: int
    message: str


# --------------------------------------------------------------------------- #
#  Query                                                                       #
# --------------------------------------------------------------------------- #

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters e.g. {'file_type': 'pdf'}"
    )
    top_k_retrieve: int = Field(default=15, ge=1, le=100)
    top_k_rerank: int = Field(default=5, ge=1, le=20)
    use_7b: bool = Field(
        default=False,
        description="Use the larger 7B model for this request"
    )


class RetrievalMeta(BaseModel):
    candidates_retrieved: int
    chunks_to_llm: int
    filters_applied: Optional[Dict[str, Any]]


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: List[str]
    model: str
    chunks_used: int
    retrieval: RetrievalMeta


# --------------------------------------------------------------------------- #
#  Health                                                                      #
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str                  # "ok" | "degraded"
    ollama_reachable: bool
    dense_chunks: int
    sparse_chunks: int
    version: str