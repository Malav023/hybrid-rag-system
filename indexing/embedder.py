from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        BGE models need a query prefix at inference time.
        For indexing (documents), no prefix is used.
        For querying, use embed_query() instead.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,   # cosine similarity ready
            batch_size=32,
            show_progress_bar=len(texts) > 50
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        BGE requires 'Represent this sentence for searching relevant passages: '
        prefix on the QUERY side only, not the document side.
        """
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True
        )
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()