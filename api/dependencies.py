"""
Shared FastAPI dependencies.

- get_rag_pipeline : injects the singleton RAGPipeline from app.state
- verify_api_key   : header-based API key guard (X-API-Key)
"""

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from config.settings import settings

# --------------------------------------------------------------------------- #
#  API Key Auth                                                                #
# --------------------------------------------------------------------------- #

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """
    Reject requests that don't carry the correct API key.
    If API_KEY is not set in .env, auth is disabled (dev mode).
    Logs a warning so you don't forget to set it in production.
    """
    expected = getattr(settings, "API_KEY", None)
    if not expected:
        # Auth disabled — warn loudly so it's never silently left open in prod
        import logging
        logging.getLogger(__name__).warning(
            "API_KEY is not set. Authentication is DISABLED. "
            "Set API_KEY in .env before deploying to production."
        )
        return
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# --------------------------------------------------------------------------- #
#  Pipeline Injection                                                          #
# --------------------------------------------------------------------------- #

def get_rag_pipeline(request: Request):
    """Inject the singleton RAGPipeline initialised at startup."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized. Server may still be starting.",
        )
    return pipeline


def get_dense_index(request: Request):
    idx = getattr(request.app.state, "dense_index", None)
    if idx is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dense index not available.",
        )
    return idx


def get_sparse_index(request: Request):
    idx = getattr(request.app.state, "sparse_index", None)
    if idx is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sparse index not available.",
        )
    return idx