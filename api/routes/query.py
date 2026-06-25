import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_rag_pipeline, verify_api_key
from api.schemas import QueryRequest, QueryResponse, RetrievalMeta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG pipeline",
    dependencies=[Depends(verify_api_key)],
)
def query_rag(
    request: QueryRequest,
    rag_pipeline=Depends(get_rag_pipeline),
):
    """
    Submit a natural language question against the indexed documents.

    The pipeline runs:
    1. Hybrid retrieval (dense + sparse → RRF fusion)
    2. CrossEncoder reranking
    3. Local LLM generation (grounded, no hallucination)

    If the answer is not in the indexed context, `grounded` will be `false`
    and `answer` will be `"NOT_IN_CONTEXT"`.
    """
    logger.info(
        f"Query received | q='{request.question[:80]}' "
        f"top_k_retrieve={request.top_k_retrieve} "
        f"top_k_rerank={request.top_k_rerank} "
        f"use_7b={request.use_7b}"
    )

    try:
        result = rag_pipeline.query(
            question=request.question,
            top_k_retrieve=request.top_k_retrieve,
            top_k_rerank=request.top_k_rerank,
            filters=request.filters,
            use_7b=request.use_7b,
        )
    except ConnectionError as exc:
        # Ollama is down
        logger.error(f"Ollama unreachable: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM backend (Ollama) is not reachable. "
                   "Ensure Ollama is running: `ollama serve`",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during RAG query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc

    retrieval_raw = result.get("retrieval", {})

    return QueryResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        sources=result["sources"],
        model=result["model"],
        chunks_used=result["chunks_used"],
        retrieval=RetrievalMeta(
            candidates_retrieved=retrieval_raw.get("candidates_retrieved", 0),
            chunks_to_llm=retrieval_raw.get("chunks_to_llm", 0),
            filters_applied=retrieval_raw.get("filters_applied"),
        ),
    )