import chromadb
import logging
import uuid
from typing import List, Dict, Any, Optional
from chromadb.config import Settings as ChromaSettings

from .embedder import Embedder
from ingestion.parsers.base_parser import ParsedChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "hybrid_rag_chunks"
HEADER_COLLECTION_NAME = "hybrid_rag_headers"


class DenseIndex:
    def __init__(self, persist_dir: str = "./index_store/chroma",
                 embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.embedder = Embedder(embedding_model)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Main collection: all chunks (prose + table rows)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        # Header collection: table column headers indexed separately
        # This lets queries like "what columns does the sales table have?"
        # hit a precise vector instead of a noisy full-row chunk
        self.header_collection = self.client.get_or_create_collection(
            name=HEADER_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"DenseIndex ready — {self.collection.count()} chunks in store")

    # ------------------------------------------------------------------ #
    #  ADD                                                                 #
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: List[ParsedChunk], batch_size: int = 100) -> int:
        """
        Index chunks in batches. Table chunks also get their headers
        indexed separately into the header collection.
        """
        added = 0

        # Split into batches to avoid memory spikes with large corpora
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]

            texts, ids, metadatas = [], [], []
            header_texts, header_ids, header_metas = [], [], []

            for chunk in batch:
                chunk_id = str(uuid.uuid4())
                # Sanitize metadata — ChromaDB only allows str/int/float/bool
                meta = self._sanitize_metadata(chunk.metadata)
                meta["chunk_id"] = chunk_id

                texts.append(chunk.text)
                ids.append(chunk_id)
                metadatas.append(meta)

                # Index column headers separately for table chunks
                if (chunk.metadata.get("chunk_type") == "table_row"
                        and chunk.metadata.get("headers")):
                    headers = chunk.metadata["headers"]
                    # Build a natural language description of the table schema
                    header_text = (
                        f"Table from {chunk.metadata.get('source_file', 'unknown')} "
                        f"with columns: {', '.join(str(h) for h in headers)}"
                    )
                    header_meta = {
                        "source_file": str(chunk.metadata.get("source_file", "")),
                        "file_type": str(chunk.metadata.get("file_type", "")),
                        "sheet_name": str(chunk.metadata.get("sheet_name", "")),
                        "headers_str": ", ".join(str(h) for h in headers),
                        "chunk_type": "table_schema",
                        "parent_chunk_id": chunk_id,
                    }
                    header_texts.append(header_text)
                    header_ids.append(str(uuid.uuid4()))
                    header_metas.append(header_meta)

            # Embed and upsert main chunks
            embeddings = self.embedder.embed(texts)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            added += len(batch)

            # Embed and upsert headers (deduplicate by header_text naturally
            # since similar tables produce similar vectors — cosine space handles it)
            if header_texts:
                header_embeddings = self.embedder.embed(header_texts)
                self.header_collection.upsert(
                    ids=header_ids,
                    embeddings=header_embeddings,
                    documents=header_texts,
                    metadatas=header_metas
                )

            logger.info(f"Indexed batch {i // batch_size + 1} — "
                        f"{added}/{len(chunks)} chunks")

        return added

    # ------------------------------------------------------------------ #
    #  QUERY                                                               #
    # ------------------------------------------------------------------ #

    def query(self, query_text: str, top_k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the main chunk collection.
        filters: ChromaDB where clause e.g. {"file_type": "xlsx"}
        """
        query_embedding = self.embedder.embed_query(query_text)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"]
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)
        return self._format_results(results)

    def query_headers(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the header/schema collection.
        Useful for questions like 'which table has revenue data?'
        """
        if self.header_collection.count() == 0:
            return []
        query_embedding = self.embedder.embed_query(query_text)
        results = self.header_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.header_collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        return self._format_results(results)

    # ------------------------------------------------------------------ #
    #  UTILS                                                               #
    # ------------------------------------------------------------------ #

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        formatted = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            formatted.append({
                "id": doc_id,
                "text": docs[i],
                "metadata": metas[i],
                # ChromaDB cosine distance → similarity score
                "score": round(1 - distances[i], 4),
            })
        return formatted

    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB rejects lists and None values in metadata.
        Convert lists to comma-separated strings, drop Nones.
        """
        clean = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, list):
                clean[k] = ", ".join(str(i) for i in v)
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def delete_by_source(self, source_file: str) -> int:
        """Re-ingest a file cleanly — delete all chunks from that source first."""
        results = self.collection.get(
            where={"source_file": source_file}
        )
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for {source_file}")
        return len(ids)

    @property
    def chunk_count(self) -> int:
        return self.collection.count()

    @property
    def header_count(self) -> int:
        return self.header_collection.count()