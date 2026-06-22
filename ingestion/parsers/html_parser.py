from bs4 import BeautifulSoup
from typing import List
from .base_parser import BaseParser, ParsedChunk


class HTMLParser(BaseParser):
    def parse(self, file_path: str) -> List[ParsedChunk]:
        chunks = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        chunks.extend(self._extract_sections(soup, file_path))
        chunks.extend(self._extract_tables(soup, file_path))
        return chunks

    def _extract_sections(self, soup, file_path: str) -> List[ParsedChunk]:
        chunks = []
        headings = soup.find_all(["h1", "h2", "h3", "h4"])
        if not headings:
            text = soup.get_text(separator=" ").strip()
            if text:
                meta = self._base_metadata(file_path, "html")
                meta["chunk_type"] = "prose"
                chunks.append(ParsedChunk(text=text, metadata=meta))
            return chunks

        for heading in headings:
            section_texts = [heading.get_text(strip=True)]
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3", "h4"]:
                    break
                text = sibling.get_text(separator=" ", strip=True)
                if text:
                    section_texts.append(text)
            full_text = " ".join(section_texts)
            if not full_text.strip():
                continue
            meta = self._base_metadata(file_path, "html")
            meta.update({
                "section_heading": heading.get_text(strip=True),
                "chunk_type": "prose",
            })
            chunks.append(ParsedChunk(text=full_text, metadata=meta))
        return chunks

    def _extract_tables(self, soup, file_path: str) -> List[ParsedChunk]:
        chunks = []
        for table_idx, table in enumerate(soup.find_all("table")):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            for row_idx, row in enumerate(rows[1:], start=1):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                row_text = " | ".join(
                    f"{headers[i]}={cell}"
                    for i, cell in enumerate(cells)
                    if i < len(headers) and cell
                )
                if not row_text.strip():
                    continue
                meta = self._base_metadata(file_path, "html")
                meta.update({
                    "table_index": table_idx,
                    "row_index": row_idx,
                    "headers": headers,
                    "chunk_type": "table_row",
                })
                chunks.append(ParsedChunk(text=row_text, metadata=meta))
        return chunks