import openpyxl
import pandas as pd
from typing import List
from .base_parser import BaseParser, ParsedChunk


class XLSXParser(BaseParser):
    def __init__(self, row_batch_size: int = 5):
        # Batch multiple rows together for better semantic context
        self.row_batch_size = row_batch_size

    def parse(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        wb = openpyxl.load_workbook(file_path, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            df = pd.DataFrame(ws.values)
            if df.empty:
                continue
            df.columns = df.iloc[0]  # first row as headers
            df = df[1:].reset_index(drop=True)
            df.dropna(how="all", inplace=True)
            df.columns = [
                str(c).strip() if c else f"col_{i}"
                for i, c in enumerate(df.columns)
            ]
            chunks.extend(
                self._rows_to_chunks(df, file_path, sheet_name)
            )
        return chunks

    def _rows_to_chunks(self, df: pd.DataFrame, file_path: str,
                         sheet_name: str) -> List[ParsedChunk]:
        chunks = []
        headers = list(df.columns)
        rows = df.to_dict(orient="records")

        for batch_start in range(0, len(rows), self.row_batch_size):
            batch = rows[batch_start: batch_start + self.row_batch_size]
            lines = []
            for row in batch:
                line = " | ".join(
                    f"{k}={v}" for k, v in row.items()
                    if v is not None and str(v).strip()
                )
                if line:
                    lines.append(line)

            if not lines:
                continue

            text = "\n".join(lines)
            meta = self._base_metadata(file_path, "xlsx")
            meta.update({
                "sheet_name": sheet_name,
                "row_start": batch_start + 1,
                "row_end": batch_start + len(batch),
                "headers": headers,
                "chunk_type": "table_row",
            })
            chunks.append(ParsedChunk(text=text, metadata=meta))
        return chunks