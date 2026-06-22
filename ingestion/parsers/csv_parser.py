import pandas as pd
from typing import List
from .base_parser import BaseParser, ParsedChunk


class CSVParser(BaseParser):
    def __init__(self, row_batch_size: int = 5):
        self.row_batch_size = row_batch_size

    def parse(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        # Try common encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode CSV: {file_path}")

        df.dropna(how="all", inplace=True)
        df.columns = [str(c).strip() for c in df.columns]
        headers = list(df.columns)
        rows = df.to_dict(orient="records")

        for batch_start in range(0, len(rows), self.row_batch_size):
            batch = rows[batch_start: batch_start + self.row_batch_size]
            lines = []
            for row in batch:
                line = " | ".join(
                    f"{k}={v}" for k, v in row.items()
                    if v is not None and str(v).strip() != ""
                )
                if line:
                    lines.append(line)

            if not lines:
                continue

            text = "\n".join(lines)
            meta = self._base_metadata(file_path, "csv")
            meta.update({
                "row_start": batch_start + 1,
                "row_end": batch_start + len(batch),
                "headers": headers,
                "chunk_type": "table_row",
            })
            chunks.append(ParsedChunk(text=text, metadata=meta))

        return chunks