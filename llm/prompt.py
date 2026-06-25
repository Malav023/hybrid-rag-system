from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer the question using ONLY the information in the context chunks below.
Rules:
- Reply with a concise, direct answer only.
- Do NOT repeat or quote the source headers (e.g. "[1] Source: ...").
- Do NOT restate the context or the question.
- If the context does not contain the answer, reply with exactly: NOT_IN_CONTEXT
- Do not use any outside knowledge."""

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

    user_message = (
        f"Context:\n{context_block}\n"
        f"Question: {query}\n"
        f"Answer (use ONLY the context, or reply NOT_IN_CONTEXT):"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]