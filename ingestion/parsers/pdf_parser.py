import fitz  # PyMuPDF
import pdfplumber
from typing import List
from .base_parser import BaseParser, ParsedChunk


class PDFParser(BaseParser):
    def parse(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        chunks.extend(self._extract_text(file_path))
        chunks.extend(self._extract_tables(file_path))
        return chunks

    def _extract_text(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
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
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    headers = [str(h).strip() if h else f"col_{i}"
                               for i, h in enumerate(table[0])]
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
        return chunks