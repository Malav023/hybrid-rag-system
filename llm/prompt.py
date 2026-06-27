from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a document QA assistant. Answer the user's question using ONLY the context provided below.

RULES:
1. Answer concisely — two to four sentences maximum.
2. Use ONLY facts present in the provided context chunks.
3. If the question asks what the documents are about, give a brief summary of the topics covered in the context.
4. If the context genuinely does not contain enough information to answer, reply with exactly: NOT_IN_CONTEXT
5. Do NOT invent facts, repeat the context verbatim, or generate follow-up questions.
6. Do NOT include preamble like "Based on the context..." — go straight to the answer."""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source_file", "unknown")
        chunk_type = meta.get("chunk_type", "unknown")

        locator_parts = []
        if meta.get("page_number"):
            locator_parts.append(f"page {meta['page_number']}")
        if meta.get("sheet_name"):
            locator_parts.append(f"sheet '{meta['sheet_name']}'")
        if meta.get("slide_number"):
            locator_parts.append(f"slide {meta['slide_number']}")
        if meta.get("section_heading"):
            locator_parts.append(f"section '{meta['section_heading']}'")

        locator = f" ({', '.join(locator_parts)})" if locator_parts else ""
        header = f"[{i}] Source: {source}{locator} | Type: {chunk_type}"
        lines.append(header)
        lines.append(chunk["text"])
        lines.append("")

    return "\n".join(lines)


def build_messages(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_block = build_context_block(chunks)

    user_content = f"""{SYSTEM_PROMPT}

---
CONTEXT:
{context_block}
---

QUESTION: {query}

ANSWER:"""

    return [
        {"role": "user", "content": user_content},
    ]