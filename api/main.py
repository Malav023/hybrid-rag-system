"""
FastAPI application entry point.

Startup sequence:
  1. Create UPLOAD_DIR and INDEX_DIR if missing
  2. Instantiate DenseIndex + SparseIndex (loads existing persisted data)
  3. Instantiate RAGPipeline (loads embedder + reranker models)
  4. Attach all three to app.state for dependency injection
  5. Serve

Shutdown sequence:
  - Nothing explicit needed; ChromaDB + BM25 persist on every write.
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import ingest_router, query_router
from api.schemas import HealthResponse
from config.settings import settings
from indexing import DenseIndex, SparseIndex
from llm import RAGPipeline

# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Lifespan — startup / shutdown                                               #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    logger.info("=== Hybrid RAG API starting up ===")

    # Ensure required directories exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"UPLOAD_DIR : {settings.UPLOAD_DIR}")
    logger.info(f"INDEX_DIR  : {settings.INDEX_DIR}")

    # Instantiate indexes (will load existing persisted data if present)
    logger.info("Loading DenseIndex ...")
    dense_index = DenseIndex(
        persist_dir=str(settings.INDEX_DIR / "chroma"),
        embedding_model=settings.EMBEDDING_MODEL,
    )
    logger.info(f"DenseIndex ready — {dense_index.chunk_count} chunks")

    logger.info("Loading SparseIndex ...")
    sparse_index = SparseIndex(
        persist_dir=str(settings.INDEX_DIR / "bm25"),
    )
    logger.info(f"SparseIndex ready — {sparse_index.chunk_count} chunks")

    # Instantiate the full RAG pipeline (loads CrossEncoder + Ollama client)
    logger.info("Building RAGPipeline ...")
    rag_pipeline = RAGPipeline(dense_index=dense_index, sparse_index=sparse_index)
    logger.info("RAGPipeline ready")

    # Ollama health check — warn but don't crash; Ollama might start later
    if rag_pipeline.llm.health_check():
        logger.info("Ollama reachable ✓")
    else:
        logger.warning(
            "Ollama is NOT reachable at startup. "
            "Query endpoints will return 503 until Ollama is running."
        )

    # Attach to app.state for dependency injection
    app.state.dense_index  = dense_index
    app.state.sparse_index = sparse_index
    app.state.rag_pipeline = rag_pipeline

    logger.info("=== Startup complete — serving requests ===")

    yield   # <-- app runs here

    # ---- SHUTDOWN ----
    logger.info("=== Hybrid RAG API shutting down ===")
    # ChromaDB and BM25 persist on every write — no explicit flush needed


# --------------------------------------------------------------------------- #
#  App factory                                                                 #
# --------------------------------------------------------------------------- #

def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid RAG API",
        description=(
            "Production-grade hybrid retrieval-augmented generation API. "
            "Supports ingestion of PDF, DOCX, XLSX, CSV, PPTX, HTML, TXT "
            "and answers questions grounded strictly in the indexed documents."
        ),
        version="0.1.0",
        lifespan=lifespan,
        # Disable docs in production via env; keep on for dev
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — tighten allowed_origins before going to production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Global unhandled exception handler — never leak stack traces to clients
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.method} {request.url}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    # Routers
    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(query_router,  prefix="/api/v1")

    # Health endpoint (no auth — needed by load balancers / k8s probes)
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Liveness + readiness probe",
    )
    def health(request: Request):
        dense  = getattr(request.app.state, "dense_index",  None)
        sparse = getattr(request.app.state, "sparse_index", None)
        rag    = getattr(request.app.state, "rag_pipeline", None)

        ollama_ok     = rag.llm.health_check() if rag else False
        dense_chunks  = dense.chunk_count  if dense  else -1
        sparse_chunks = sparse.chunk_count if sparse else -1

        overall_status = "ok" if ollama_ok else "degraded"

        return HealthResponse(
            status=overall_status,
            ollama_reachable=ollama_ok,
            dense_chunks=dense_chunks,
            sparse_chunks=sparse_chunks,
            version="0.1.0",
        )

    return app


app = create_app()


# --------------------------------------------------------------------------- #
#  Dev entrypoint                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,        # Never True in production
        log_level="info",
    )