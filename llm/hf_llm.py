"""
Hugging Face Inference API LLM client.
Drop-in replacement for GroqLLM / OllamaLLM.

Free models that work without gating:
  - HuggingFaceH4/zephyr-7b-beta          (recommended, no approval needed)
  - mistralai/Mistral-7B-Instruct-v0.3    (no approval needed)
  - microsoft/Phi-3-mini-4k-instruct      (lightweight, fast)
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from llm.prompt import build_messages
from config.settings import settings

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/{model}/v1/chat/completions"


class HuggingFaceLLM:

    def __init__(self):
        self.hf_token = settings.HF_TOKEN
        self.model = settings.HF_MODEL
        self.max_tokens = settings.HF_MAX_TOKENS
        self.temperature = settings.HF_TEMPERATURE

        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN is not set. Add it as a secret in your HF Space settings."
            )

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

        logger.info(f"Sending query to HF Inference [{model}] — {len(chunks)} chunks in context")

        try:
            raw_answer = self._post(payload, model)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error(f"HF API HTTP {exc.code}: {body}")
            if exc.code == 401:
                raise ConnectionError("HF_TOKEN is invalid or expired.") from exc
            if exc.code == 403:
                raise ConnectionError(
                    f"Model '{model}' is gated or requires approval. "
                    "Switch HF_MODEL to HuggingFaceH4/zephyr-7b-beta in Space secrets."
                ) from exc
            if exc.code == 503:
                raise ConnectionError(
                    "HF model is loading (cold start). Retry in 20-30 seconds."
                ) from exc
            raise ConnectionError(f"HF API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            logger.error(f"HF API connection failed: {exc}")
            raise ConnectionError(f"Cannot reach HF Inference API: {exc}") from exc

        raw_answer = raw_answer.strip()
        grounded = raw_answer != "NOT_IN_CONTEXT"
        sources = self._extract_sources(chunks) if grounded else []

        logger.info(f"HF response — grounded={grounded}, length={len(raw_answer)}")

        return {
            "answer":      raw_answer,
            "sources":     sources,
            "model":       model,
            "chunks_used": len(chunks),
            "grounded":    grounded,
        }

    def health_check(self) -> bool:
        if not self.hf_token:
            return False
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            }
            self._post(payload, self.model)
            return True
        except urllib.error.HTTPError as exc:
            # 503 = model cold start — token is valid, treat as healthy
            if exc.code == 503:
                logger.warning("HF model cold start (503) — treating as healthy.")
                return True
            logger.error(f"HF health check failed: HTTP {exc.code}")
            return False
        except Exception as exc:
            logger.error(f"HF health check failed: {exc}")
            return False

    def _post(self, payload: Dict[str, Any], model: str) -> str:
        url = HF_API_URL.format(model=model)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.hf_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
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

    def _not_in_context(self, query: str, model: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "answer":      "NOT_IN_CONTEXT",
            "sources":     [],
            "model":       model,
            "chunks_used": len(chunks),
            "grounded":    False,
        }