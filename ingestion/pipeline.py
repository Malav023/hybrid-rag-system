import os
import logging
from typing import List, Dict, Type, Optional, TYPE_CHECKING
from .parsers.base_parser import BaseParser, ParsedChunk
from .parsers import (
    PDFParser, DOCXParser, XLSXParser,
    CSVParser, PPTXParser, HTMLParser, TXTParser
)
from .chunker import SemanticChunker

if TYPE_CHECKING:
    from indexing.dense_index import DenseIndex
    from indexing.sparse_index import SparseIndex

logger = logging.getLogger(__name__)

PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {
    ".pdf":  PDFParser,
    ".docx": DOCXParser,
    ".xlsx": XLSXParser,
    ".xls":  XLSXParser,
    ".csv":  CSVParser,
    ".pptx": PPTXParser,
    ".ppt":  PPTXParser,
    ".html": HTMLParser,
    ".htm":  HTMLParser,
    ".txt":  TXTParser,
    ".md":   TXTParser,
}


class IngestionPipeline:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 table_row_batch_size: int = 5):
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.table_row_batch_size = table_row_batch_size

    def ingest_file(
        self,
        file_path: str,
        dense_index: Optional["DenseIndex"] = None,
        sparse_index: Optional["SparseIndex"] = None,
    ) -> List[ParsedChunk]:
        """
        Parse and chunk a file.

        If dense_index or sparse_index are provided, this method will
        automatically delete any existing chunks from that source file
        before adding the new ones — preventing duplication on re-ingest.

        This is the correct production call signature. The indexes are
        optional so the pipeline can also be used standalone (e.g. in tests
        or batch jobs that manage indexing separately).
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext not in PARSER_REGISTRY:
            raise ValueError(f"Unsupported file type: {ext}")

        parser_cls = PARSER_REGISTRY[ext]

        if ext in (".xlsx", ".xls", ".csv"):
            parser = parser_cls(row_batch_size=self.table_row_batch_size)
        else:
            parser = parser_cls()

        source_filename = os.path.basename(file_path)

        # --- Deduplication: purge existing chunks for this file first ---
        if dense_index is not None:
            removed = dense_index.delete_by_source(source_filename)
            if removed:
                logger.info(f"Removed {removed} stale dense chunks for '{source_filename}'")

        if sparse_index is not None:
            removed = sparse_index.delete_by_source(source_filename)
            if removed:
                logger.info(f"Removed {removed} stale sparse chunks for '{source_filename}'")

        logger.info(f"Parsing {file_path} with {parser_cls.__name__}")
        raw_chunks = parser.parse(file_path)
        logger.info(f"  → {len(raw_chunks)} raw chunks extracted")

        final_chunks = self.chunker.chunk(raw_chunks)
        logger.info(f"  → {len(final_chunks)} chunks after semantic chunking")

        # --- Index immediately if indexes are provided ---
        if dense_index is not None:
            dense_index.add_chunks(final_chunks)

        if sparse_index is not None:
            sparse_index.add_chunks(final_chunks)

        return final_chunks

    def ingest_directory(
        self,
        dir_path: str,
        dense_index: Optional["DenseIndex"] = None,
        sparse_index: Optional["SparseIndex"] = None,
    ) -> List[ParsedChunk]:
        """
        Ingest all supported files in a directory.
        Passes indexes down to ingest_file so each file is deduplicated.
        """
        all_chunks = []
        for root, _, files in os.walk(dir_path):
            for fname in files:
                ext = os.path.splitext(fname)[-1].lower()
                if ext not in PARSER_REGISTRY:
                    continue
                full_path = os.path.join(root, fname)
                try:
                    chunks = self.ingest_file(
                        full_path,
                        dense_index=dense_index,
                        sparse_index=sparse_index,
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to parse {full_path}: {e}")
        return all_chunks