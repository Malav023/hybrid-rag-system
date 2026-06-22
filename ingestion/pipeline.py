import os
import logging
from typing import List, Dict, Type
from .parsers.base_parser import BaseParser, ParsedChunk
from .parsers import (
    PDFParser, DOCXParser, XLSXParser,
    CSVParser, PPTXParser, HTMLParser
)
from .chunker import SemanticChunker

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
}


class IngestionPipeline:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 table_row_batch_size: int = 5):
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.table_row_batch_size = table_row_batch_size

    def ingest_file(self, file_path: str) -> List[ParsedChunk]:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext not in PARSER_REGISTRY:
            raise ValueError(f"Unsupported file type: {ext}")

        parser_cls = PARSER_REGISTRY[ext]

        # Inject batch size for tabular parsers
        if ext in (".xlsx", ".xls", ".csv"):
            parser = parser_cls(row_batch_size=self.table_row_batch_size)
        else:
            parser = parser_cls()

        logger.info(f"Parsing {file_path} with {parser_cls.__name__}")
        raw_chunks = parser.parse(file_path)
        logger.info(f"  → {len(raw_chunks)} raw chunks extracted")

        final_chunks = self.chunker.chunk(raw_chunks)
        logger.info(f"  → {len(final_chunks)} chunks after semantic chunking")

        return final_chunks

    def ingest_directory(self, dir_path: str) -> List[ParsedChunk]:
        all_chunks = []
        for root, _, files in os.walk(dir_path):
            for fname in files:
                ext = os.path.splitext(fname)[-1].lower()
                if ext not in PARSER_REGISTRY:
                    continue
                full_path = os.path.join(root, fname)
                try:
                    chunks = self.ingest_file(full_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to parse {full_path}: {e}")
        return all_chunks