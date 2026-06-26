"""
Hugging Face Inference API LLM client.
Uses huggingface_hub InferenceClient which works inside HF Spaces
via internal routing (no external DNS needed).

Free models (no approval required):
  - HuggingFaceH4/zephyr-7b-beta
  - mistralai/Mistral-7B-Instruct-v0.3
  - microsoft/Phi-3-mini-4k-instruct
"""

import logging
from typing import Any, Dict, List, Optional

from llm.prompt import build_messages
from config.settings import settings

logger = logging.getLogger(__name__)


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

        # Import here so startup fails fast if huggingface_hub is missing
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(
            model=self.model,
            token=self.hf_token,
        )
        logger.info(f"HuggingFaceLLM ready — model: {self.model}")

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

        logger.info(
            f"Sending query to HF Inference [{model}] — "
            f"{len(chunks)} chunks in context"
        )

        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=model, token=self.hf_token)

            response = client.chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw_answer = response.choices[0].message.content.strip()

        except Exception as exc:
            logger.error(f"HF inference failed: {exc}")
            raise ConnectionError(f"HF Inference API error: {exc}") from exc

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
        """
        Use a simple token check instead of a live inference call.
        Avoids cold-start delays at startup.
        """
        if not self.hf_token:
            return False
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            api.whoami()   # lightweight auth check, no model call
            logger.info("HF token verified via whoami()")
            return True
        except Exception as exc:
            logger.error(f"HF health check failed: {exc}")
            return False

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