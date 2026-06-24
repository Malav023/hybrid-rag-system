from typing import List
from .base_parser import BaseParser, ParsedChunk


class TXTParser(BaseParser):
    def parse(self, file_path: str) -> List[ParsedChunk]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if not text:
            return []
        meta = self._base_metadata(file_path, "txt")
        meta["chunk_type"] = "prose"
        return [ParsedChunk(text=text, metadata=meta)]