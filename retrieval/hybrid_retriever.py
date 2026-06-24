import logging
from typing import List, Dict, Any, Optional

from indexing.dense_index import DenseIndex
from indexing.sparse_index import SparseIndex

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60          # standard constant — don't tune without reason
DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_SPARSE_WEIGHT = 0.4


class HybridRetriever:
    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
    ):
        self.dense = dense_index
        self.sparse = sparse_index
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    # ------------------------------------------------------------------ #
    #  PUBLIC                                                              #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_schema: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval via RRF fusion.

        Args:
            query:          Natural language query string.
            top_k:          Number of final results to return.
            filters:        Metadata filter dict passed to both indexes.
                            e.g. {"file_type": "xlsx"}
            include_schema: Whether to also query the header collection
                            and prepend schema hits (useful for table queries).

        Returns:
            List of result dicts with keys:
                id, text, metadata, rrf_score, dense_rank, sparse_rank
        """
        fetch_k = max(top_k * 3, 30)  # over-fetch for better fusion coverage

        # --- Parallel retrieval from both indexes ---
        dense_results = self.dense.query(query, top_k=fetch_k, filters=filters)
        sparse_results = self.sparse.query(query, top_k=fetch_k, filters=filters)

        logger.info(
            f"Pre-fusion: dense={len(dense_results)}, sparse={len(sparse_results)}"
        )

        # --- RRF fusion ---
        fused = self._rrf_fuse(dense_results, sparse_results)

        # --- Optional: prepend schema/header hits for table-schema queries ---
        if include_schema and self._looks_like_schema_query(query):
            schema_hits = self.dense.query_headers(query, top_k=3)
            # Prepend schema hits (deduplicated) before the fused prose/row results
            seen_texts = {self._fingerprint(r["text"]) for r in fused}
            for hit in schema_hits:
                if self._fingerprint(hit["text"]) not in seen_texts:
                    hit["rrf_score"] = 1.0   # treat schema hits as top-priority
                    hit["source"] = "schema"
                    fused.insert(0, hit)

        return fused[:top_k]

    # ------------------------------------------------------------------ #
    #  RRF CORE                                                            #
    # ------------------------------------------------------------------ #

    def _rrf_fuse(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion over dense + sparse result lists.
        Each list is already ranked (index 0 = best).
        """
        scores: Dict[str, Dict[str, Any]] = {}  # fingerprint → merged entry

        # Score dense results
        for rank, result in enumerate(dense_results):
            fp = self._fingerprint(result["text"])
            rrf_contrib = self.dense_weight / (self.rrf_k + rank + 1)
            if fp not in scores:
                scores[fp] = {**result, "rrf_score": 0.0,
                              "dense_rank": None, "sparse_rank": None,
                              "source": "dense"}
            scores[fp]["rrf_score"] += rrf_contrib
            scores[fp]["dense_rank"] = rank + 1

        # Score sparse results
        for rank, result in enumerate(sparse_results):
            fp = self._fingerprint(result["text"])
            rrf_contrib = self.sparse_weight / (self.rrf_k + rank + 1)
            if fp not in scores:
                scores[fp] = {**result, "rrf_score": 0.0,
                              "dense_rank": None, "sparse_rank": None,
                              "source": "sparse"}
            else:
                scores[fp]["source"] = "both"   # appeared in both — good signal
            scores[fp]["rrf_score"] += rrf_contrib
            scores[fp]["sparse_rank"] = rank + 1

        # Sort by fused RRF score descending
        fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        logger.info(f"Post-fusion: {len(fused)} unique chunks")
        return fused

    # ------------------------------------------------------------------ #
    #  UTILS                                                               #
    # ------------------------------------------------------------------ #

    def _fingerprint(self, text: str) -> str:
        """
        Cheap deduplication key: first 120 chars, lowercased, whitespace-normalized.
        Avoids hashing overhead while being collision-resistant enough for RAG.
        """
        return " ".join(text.lower().split())[:120]

    def _looks_like_schema_query(self, query: str) -> bool:
        """
        Heuristic to detect schema/column-oriented questions.
        Triggers the header collection lookup.
        """
        schema_signals = [
            "column", "columns", "field", "fields", "schema",
            "header", "headers", "what data", "what information",
            "what does the table", "attributes", "structure"
        ]
        q_lower = query.lower()
        return any(signal in q_lower for signal in schema_signals)