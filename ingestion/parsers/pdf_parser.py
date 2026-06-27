import fitz  # PyMuPDF
import pdfplumber
import logging
from typing import List
from .base_parser import BaseParser, ParsedChunk

logger = logging.getLogger(__name__)

PDFPLUMBER_PAGE_LIMIT = 100


class PDFParser(BaseParser):
    def parse(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        chunks.extend(self._extract_text(file_path))

        page_count = self._page_count(file_path)
        if page_count <= PDFPLUMBER_PAGE_LIMIT:
            chunks.extend(self._extract_tables(file_path))
        else:
            logger.warning(
                f"PDF has {page_count} pages — skipping pdfplumber table extraction "
                f"(limit={PDFPLUMBER_PAGE_LIMIT}). Only prose text will be indexed."
            )

        return chunks

    def _page_count(self, file_path: str) -> int:
        try:
            doc = fitz.open(file_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception:
            return 0

    def _extract_text(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            logger.error(f"fitz failed to open {file_path}: {exc}")
            return chunks

        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text").strip()
            except Exception as exc:
                logger.warning(f"fitz failed on page {page_num + 1}: {exc}")
                continue

            if not text:
                continue

            meta = self._base_metadata(file_path, "pdf")
            meta.update({
                "page_number": page_num + 1,
                "chunk_type": "prose",
            })
            chunks.append(ParsedChunk(text=text, metadata=meta))

        doc.close()
        return chunks

    def _extract_tables(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        tables = page.extract_tables()
                    except Exception as exc:
                        logger.warning(f"pdfplumber failed on page {page_num + 1}: {exc}")
                        continue

                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        headers = [
                            str(h).strip() if h else f"col_{i}"
                            for i, h in enumerate(table[0])
                        ]
                        for row_idx, row in enumerate(table[1:], start=1):
                            row_text = " | ".join(
                                f"{headers[i]}={str(cell).strip()}"
                                for i, cell in enumerate(row)
                                if cell is not None
                            )
                            if not row_text.strip():
                                continue
                            meta = self._base_metadata(file_path, "pdf")
                            meta.update({
                                "page_number": page_num + 1,
                                "table_index": table_idx,
                                "row_index": row_idx,
                                "headers": headers,
                                "chunk_type": "table_row",
                            })
                            chunks.append(ParsedChunk(text=row_text, metadata=meta))
        except Exception as exc:
            logger.error(f"pdfplumber failed entirely on {file_path}: {exc}")

        return chunks