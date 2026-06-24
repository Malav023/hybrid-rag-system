import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 32


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        logger.info(f"Loading CrossEncoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    #  PUBLIC                                                              #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-score a list of retrieval results using a CrossEncoder.

        The CrossEncoder sees the full (query, passage) pair jointly —
        far more accurate than bi-encoder cosine similarity, but too slow
        to run over the whole corpus. So we run it only on the small
        candidate set that comes out of the hybrid retriever (typically 10-30).

        Args:
            query:   Original user query string.
            results: Output from HybridRetriever.retrieve().
            top_k:   How many results to return after re-ranking.

        Returns:
            Same dicts as input, with an added `rerank_score` field,
            sorted by that score descending.
        """
        if not results:
            return []

        # CrossEncoder expects List[Tuple[query, passage]]
        pairs = [(query, r["text"]) for r in results]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        for result, score in zip(results, scores):
            result["rerank_score"] = round(float(score), 4)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranked {len(results)} → returning top {min(top_k, len(reranked))}"
        )
        return reranked[:top_k]