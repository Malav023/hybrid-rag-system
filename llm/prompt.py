from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a strict document QA assistant. Your only job is to answer the question using the context provided. 

STRICT RULES:
- Answer in one or two sentences maximum.
- Use ONLY information from the context below.
- If the context does not contain the answer, reply with exactly the word: NOT_IN_CONTEXT
- Do NOT generate more questions.
- Do NOT repeat the context.
- Do NOT add explanations.
- Just answer."""


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

    # For Zephyr and similar models, fold system prompt into user turn
    # This prevents the model from treating the exchange as a few-shot template
    combined_user_message = f"""{SYSTEM_PROMPT}

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER (one or two sentences only, or NOT_IN_CONTEXT):"""

    return [
        {"role": "user", "content": combined_user_message},
        # Prime the assistant turn so model completes from here, not generates Q&A
        {"role": "assistant", "content": ""},
    ]