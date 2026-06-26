import logging
from typing import Any, Dict, Optional

from indexing import DenseIndex, SparseIndex
from retrieval import HybridRetriever, Reranker
from config.settings import settings

logger = logging.getLogger(__name__)


def _build_llm():
    backend = settings.LLM_BACKEND.lower()

    if backend == "huggingface":
        if not settings.HF_TOKEN:
            logger.warning(
                "LLM_BACKEND=huggingface but HF_TOKEN is not set. "
                "Falling back to Ollama."
            )
            backend = "ollama"
        else:
            from llm.hf_llm import HuggingFaceLLM
            logger.info("LLM backend: HuggingFace Inference API")
            return HuggingFaceLLM()

    if backend == "groq":
        if not settings.GROQ_API_KEY:
            logger.warning(
                "LLM_BACKEND=groq but GROQ_API_KEY is not set. "
                "Falling back to Ollama."
            )
            backend = "ollama"
        else:
            from llm.groq_llm import GroqLLM
            logger.info("LLM backend: Groq")
            return GroqLLM()

    if backend == "ollama":
        from llm.local_llm import OllamaLLM
        logger.info("LLM backend: Ollama")
        return OllamaLLM()

    raise ValueError(
        f"Unknown LLM_BACKEND='{settings.LLM_BACKEND}'. "
        f"Must be 'groq', 'ollama', or 'huggingface'."
    )


class RAGPipeline:
    """
    Assembles retriever → reranker → LLM into one callable.
    Instantiate once at app startup and reuse across requests.
    """

    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
    ):
        self.retriever = HybridRetriever(dense_index, sparse_index)
        self.reranker  = Reranker()
        self.llm       = _build_llm()

    def query(
        self,
        question: str,
        top_k_retrieve: int = settings.RETRIEVER_TOP_K,
        top_k_rerank: int   = settings.RERANKER_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        use_7b: bool = False,
    ) -> Dict[str, Any]:
        """
        Full RAG query: retrieve → rerank → generate.

        use_7b is ignored when backend is Groq (model is controlled
        by GROQ_MODEL in settings). It remains in the signature for
        API backwards compatibility.
        """
        # Step 1 — Hybrid retrieval
        candidates = self.retriever.retrieve(
            query=question,
            top_k=top_k_retrieve,
            filters=filters,
        )
        logger.info(f"Retrieved {len(candidates)} candidates")

        # Step 2 — CrossEncoder reranking
        reranked = self.reranker.rerank(
            query=question,
            results=candidates,
            top_k=top_k_rerank,
        )
        logger.info(f"Reranked to {len(reranked)} chunks")

        # Step 3 — LLM generation
        # model_override only applies to Ollama backend
        model_override = None
        if use_7b and settings.LLM_BACKEND == "ollama":
            model_override = settings.OLLAMA_MODEL_7B

        result = self.llm.answer(
            query=question,
            chunks=reranked,
            model_override=model_override,
        )

        result["retrieval"] = {
            "candidates_retrieved": len(candidates),
            "chunks_to_llm":        len(reranked),
            "filters_applied":      filters,
        }

        return result