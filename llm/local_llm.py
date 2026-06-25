import json
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from .prompt import build_messages
from config.settings import settings

logger = logging.getLogger(__name__)

class OllamaLLM:
    """
    Thin, dependency-free client for the Ollama /api/chat endpoint.
    Uses only stdlib urllib — no httpx/requests dependency needed.

    Why not the ollama Python SDK?
    The SDK is fine but adds a dependency and hides the HTTP contract.
    For a production system you want to own the retry/timeout/error
    handling explicitly, which is cleaner with raw urllib.
    """

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout: int = settings.OLLAMA_TIMEOUT,
        max_tokens: int = settings.OLLAMA_MAX_TOKENS,
        temperature: float = settings.OLLAMA_TEMPERATURE,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._chat_url = f"{self.base_url}/api/chat"

    # ------------------------------------------------------------------ #
    #  PUBLIC                                                              #
    # ------------------------------------------------------------------ #

    def answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer from the provided chunks.

        Args:
            query:          The user's question.
            chunks:         Reranked retrieval results (from Reranker.rerank()).
            model_override: Pass settings.OLLAMA_MODEL_7B to use 7b for this call.

        Returns:
            {
                "answer":   str,        # LLM response or "NOT_IN_CONTEXT"
                "sources":  list,       # deduplicated source file names cited
                "model":    str,        # model that answered
                "chunks_used": int,     # how many chunks were in the prompt
                "grounded": bool,       # False if answer is NOT_IN_CONTEXT
            }
        """
        if not chunks:
            logger.warning("No chunks provided — returning NOT_IN_CONTEXT")
            return self._not_in_context(query, model_override or self.model, chunks)

        model = model_override or self.model
        messages = build_messages(query, chunks)

        payload = {
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options": {
                "temperature":   self.temperature,
                "num_predict":   self.max_tokens,
                "stop":          ["</s>", "[INST]", "[/INST]"],
            },
        }

        logger.info(f"Sending query to Ollama [{model}] — {len(chunks)} chunks in context")

        try:
            response_text = self._post(payload)
        except urllib.error.URLError as e:
            logger.error(f"Ollama connection failed: {e}")
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is Ollama running? Run: ollama serve"
            ) from e

        raw_answer = response_text.strip()

        grounded = raw_answer != "NOT_IN_CONTEXT"
        sources = self._extract_sources(chunks) if grounded else []

        logger.info(f"Ollama response — grounded={grounded}, length={len(raw_answer)}")

        return {
            "answer":      raw_answer,
            "sources":     sources,
            "model":       model,
            "chunks_used": len(chunks),
            "grounded":    grounded,
        }

    def health_check(self) -> bool:
        """Ping Ollama to verify it's reachable before serving requests."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  PRIVATE                                                             #
    # ------------------------------------------------------------------ #

    def _post(self, payload: Dict[str, Any]) -> str:
        """POST JSON payload to Ollama /api/chat, return the answer text."""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Ollama /api/chat non-streaming response shape:
        # {"message": {"role": "assistant", "content": "..."}, ...}
        return data["message"]["content"]

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Return sources for chunks that meaningfully contributed to the answer.
        Uses rerank score gap: chunks within 2.0 of the top score are included.
        This avoids listing every chunk passed to the LLM as a source.
        """
        if not chunks:
            return []

        scores = [c.get("rerank_score", 0.0) for c in chunks]
        top_score = scores[0]
        threshold = 2.0   # rerank score gap — tunable

        seen = set()
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