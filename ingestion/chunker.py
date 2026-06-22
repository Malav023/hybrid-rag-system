import re
from typing import List
from .parsers.base_parser import ParsedChunk


class SemanticChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, parsed_chunks: List[ParsedChunk]) -> List[ParsedChunk]:
        result = []
        for pc in parsed_chunks:
            # Table rows and slides stay as-is — already atomic
            if pc.metadata.get("chunk_type") in ("table_row", "slide"):
                result.append(pc)
            else:
                result.extend(self._split_prose(pc))
        return result

    def _split_prose(self, pc: ParsedChunk) -> List[ParsedChunk]:
        sentences = self._split_sentences(pc.text)
        chunks = []
        buffer = []
        buffer_len = 0
        chunk_index = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if buffer_len + word_count > self.chunk_size and buffer:
                chunk_text = " ".join(buffer)
                meta = {**pc.metadata, "chunk_index": chunk_index}
                chunks.append(ParsedChunk(text=chunk_text, metadata=meta))
                chunk_index += 1
                # Overlap: keep last N words
                overlap_text = " ".join(
                    " ".join(buffer).split()[-self.overlap:]
                )
                buffer = [overlap_text] if overlap_text else []
                buffer_len = len(overlap_text.split())

            buffer.append(sentence)
            buffer_len += word_count

        if buffer:
            chunk_text = " ".join(buffer)
            meta = {**pc.metadata, "chunk_index": chunk_index}
            chunks.append(ParsedChunk(text=chunk_text, metadata=meta))

        return chunks if chunks else [pc]

    def _split_sentences(self, text: str) -> List[str]:
        # Simple but effective sentence splitter
        text = re.sub(r'\n+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]