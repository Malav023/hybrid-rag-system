import pickle
import os
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from ingestion.parsers.base_parser import ParsedChunk

logger = logging.getLogger(__name__)

BM25_INDEX_FILE = "bm25_index.pkl"
BM25_CORPUS_FILE = "bm25_corpus.pkl"


class SparseIndex:
    def __init__(self, persist_dir: str = "./index_store/bm25"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.bm25: Optional[BM25Okapi] = None

        # We store the full chunk objects so we can return
        # text + metadata on query, just like DenseIndex does
        self.corpus: List[Dict[str, Any]] = []  # [{text, metadata}]
        self.tokenized_corpus: List[List[str]] = []

        self._load_if_exists()
        logger.info(f"SparseIndex ready — {len(self.corpus)} chunks in store")

    # ------------------------------------------------------------------ #
    #  ADD                                                                 #
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: List[ParsedChunk]) -> int:
        """
        Tokenize and add chunks to BM25 index.
        Calling this multiple times appends — does not replace.
        Call clear() first if you want a clean re-index.
        """
        new_entries = []
        new_tokenized = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            if not tokens:
                continue
            new_tokenized.append(tokens)
            new_entries.append({
                "text": chunk.text,
                "metadata": chunk.metadata
            })

        if not new_entries:
            logger.warning("No valid chunks to add to sparse index")
            return 0

        self.tokenized_corpus.extend(new_tokenized)
        self.corpus.extend(new_entries)

        # Rebuild BM25 over full corpus
        # BM25Okapi does not support incremental updates — full rebuild is correct
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._persist()
        logger.info(f"SparseIndex: {len(self.corpus)} total chunks indexed")
        return len(new_entries)

    # ------------------------------------------------------------------ #
    #  QUERY                                                               #
    # ------------------------------------------------------------------ #

    def query(self, query_text: str, top_k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        BM25 keyword retrieval.
        filters: simple exact-match dict e.g. {"file_type": "xlsx"}
                 applied AFTER scoring (BM25 has no native filter support)
        """
        if self.bm25 is None or not self.corpus:
            logger.warning("SparseIndex is empty — returning no results")
            return []

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)  # numpy array

        # Pair each corpus entry with its BM25 score
        scored = [
            (score, idx)
            for idx, score in enumerate(scores)
            if score > 0  # skip zero-score docs
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scored:
            entry = self.corpus[idx]

            # Apply metadata filters post-scoring
            if filters and not self._matches_filters(entry["metadata"], filters):
                continue

            results.append({
                "id": f"bm25_{idx}",
                "text": entry["text"],
                "metadata": entry["metadata"],
                "score": round(float(score), 4),
            })

            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------ #
    #  DELETE / CLEAR                                                      #
    # ------------------------------------------------------------------ #

    def delete_by_source(self, source_file: str) -> int:
        """
        Remove all chunks that came from a specific file.
        Triggers a full BM25 rebuild — expected for production re-ingestion.
        """
        original_len = len(self.corpus)
        filtered = [
            (entry, tokens)
            for entry, tokens in zip(self.corpus, self.tokenized_corpus)
            if entry["metadata"].get("source_file") != source_file
        ]

        if not filtered:
            self.corpus = []
            self.tokenized_corpus = []
            self.bm25 = None
        else:
            entries, tokens = zip(*filtered)
            self.corpus = list(entries)
            self.tokenized_corpus = list(tokens)
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        removed = original_len - len(self.corpus)
        self._persist()
        logger.info(f"Deleted {removed} chunks for source: {source_file}")
        return removed

    def clear(self) -> None:
        """Wipe the entire sparse index."""
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        for f in [BM25_INDEX_FILE, BM25_CORPUS_FILE]:
            path = os.path.join(self.persist_dir, f)
            if os.path.exists(path):
                os.remove(path)
        logger.info("SparseIndex cleared")

    # ------------------------------------------------------------------ #
    #  PERSIST / LOAD                                                      #
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        """
        Pickle the BM25 object and corpus to disk.
        BM25Okapi is not large — even 100k chunks stays well under 500MB.
        For millions of chunks, swap to Elasticsearch instead.
        """
        with open(os.path.join(self.persist_dir, BM25_INDEX_FILE), "wb") as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(self.persist_dir, BM25_CORPUS_FILE), "wb") as f:
            pickle.dump((self.corpus, self.tokenized_corpus), f)
        logger.info("SparseIndex persisted to disk")

    def _load_if_exists(self) -> None:
        index_path = os.path.join(self.persist_dir, BM25_INDEX_FILE)
        corpus_path = os.path.join(self.persist_dir, BM25_CORPUS_FILE)
        if os.path.exists(index_path) and os.path.exists(corpus_path):
            with open(index_path, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(corpus_path, "rb") as f:
                self.corpus, self.tokenized_corpus = pickle.load(f)
            logger.info(f"Loaded existing SparseIndex: {len(self.corpus)} chunks")

    # ------------------------------------------------------------------ #
    #  UTILS                                                               #
    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> List[str]:
        """
        Lowercase + whitespace tokenization.
        Deliberately simple — BM25 doesn't benefit much from
        stemming/lemmatization and it adds a heavy dependency (NLTK/spaCy).
        Stopword removal is intentionally skipped: BM25's IDF
        already down-weights high-frequency terms naturally.
        """
        return text.lower().split()

    def _matches_filters(self, metadata: Dict[str, Any],
                          filters: Dict[str, Any]) -> bool:
        for key, val in filters.items():
            if metadata.get(key) != val:
                return False
        return True

    @property
    def chunk_count(self) -> int:
        return len(self.corpus)