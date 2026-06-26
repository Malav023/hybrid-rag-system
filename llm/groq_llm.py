"""
Groq LLM client — drop-in replacement for OllamaLLM.

Uses Groq's OpenAI-compatible /openai/v1/chat/completions endpoint
via stdlib urllib only — no SDK dependency.

Free tier limits (as of 2025):
  llama-3.1-8b-instant : 6000 RPM, 500k tokens/day
  These are more than sufficient for a public demo.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from llm.prompt import build_messages
from config.settings import settings

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    """
    Thin Groq client with the same public interface as OllamaLLM
    so RAGPipeline needs zero changes.
    """

    def __init__(
        self,
        api_key: str = settings.GROQ_API_KEY,
        model: str = settings.GROQ_MODEL,
        max_tokens: int = settings.GROQ_MAX_TOKENS,
        temperature: float = settings.GROQ_TEMPERATURE,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to .env"
            )
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    #  PUBLIC — same interface as OllamaLLM                               #
    # ------------------------------------------------------------------ #

    def answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not chunks:
            logger.warning("No chunks provided — returning NOT_IN_CONTEXT")
            return self._not_in_context(query, model_override or self.model, chunks)

        model = model_override or self.model
        messages = build_messages(query, chunks)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        logger.info(
            f"Sending query to Groq [{model}] — {len(chunks)} chunks in context"
        )

        try:
            raw_answer = self._post(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error(f"Groq HTTP {exc.code}: {body}")
            if exc.code == 401:
                raise ConnectionError("Groq API key is invalid or expired.") from exc
            if exc.code == 429:
                raise ConnectionError(
                    "Groq rate limit exceeded. Retry after a moment."
                ) from exc
            raise ConnectionError(f"Groq API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            logger.error(f"Groq connection failed: {exc}")
            raise ConnectionError(
                f"Cannot reach Groq API: {exc}. Check your internet connection."
            ) from exc

        raw_answer = raw_answer.strip()
        grounded = raw_answer != "NOT_IN_CONTEXT"
        sources = self._extract_sources(chunks) if grounded else []

        logger.info(
            f"Groq response — grounded={grounded}, length={len(raw_answer)}"
        )

        return {
            "answer":      raw_answer,
            "sources":     sources,
            "model":       model,
            "chunks_used": len(chunks),
            "grounded":    grounded,
        }

    def health_check(self) -> bool:
        """
        Verify Groq is reachable by making a minimal inference call.
        The /models endpoint returns 403 on free tier so we use a 1-token ping instead.
        """
        if not self.api_key:
            return False
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
            }
            self._post(payload)
            return True
        except Exception as exc:
            logger.error(f"Groq health check failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    #  PRIVATE                                                             #
    # ------------------------------------------------------------------ #

    def _post(self, payload: Dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            GROQ_CHAT_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # OpenAI-compatible response shape
        return data["choices"][0]["message"]["content"]

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        if not chunks:
            return []
        scores = [c.get("rerank_score", 0.0) for c in chunks]
        top_score = scores[0]
        threshold = 2.0
        seen: set = set()
        sources = []
        for chunk, score in zip(chunks, scores):
            if top_score - score <= threshold:
                src = chunk.get("metadata", {}).get("source_file", "unknown")
                if src not in seen:
                    seen.add(src)
                    sources.append(src)
        return sources

    def _not_in_context(
        self, query: str, model: str, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "answer":      "NOT_IN_CONTEXT",
            "sources":     [],
            "model":       model,
            "chunks_used": len(chunks),
            "grounded":    False,
        }