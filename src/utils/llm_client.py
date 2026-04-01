"""
llm_client.py — HuggingFace InferenceClient wrapper with retry logic.

Provides a unified interface for calling HuggingFace hosted models
with configurable parameters and built-in exponential backoff retries.
"""

import logging
import os
from typing import Optional

from huggingface_hub import InferenceClient
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 3


# ── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """Wrapper around HuggingFace InferenceClient with retry and logging."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize the LLM client.

        Args:
            model: HuggingFace model ID or inference endpoint URL
            token: HF API token (falls back to HF_TOKEN env var)
            max_tokens: Default max tokens for generation
            temperature: Default sampling temperature
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        token = token or os.environ.get("HF_TOKEN")
        if not token:
            logger.warning(
                "No HuggingFace token provided. Set HF_TOKEN environment variable "
                "or pass token= to LLMClient. Some models may be inaccessible."
            )

        self.client = InferenceClient(model=model, token=token)
        logger.info("LLMClient initialized with model: %s", model)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            "LLM call failed (attempt %d/%d): %s. Retrying...",
            retry_state.attempt_number,
            MAX_RETRIES,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            max_tokens: Override default max tokens
            temperature: Override default temperature
            system_prompt: Optional system prompt (prepended to messages)

        Returns:
            Generated text response
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
        )

        result = response.choices[0].message.content
        logger.debug(
            "LLM response (%d chars) for %d messages",
            len(result), len(messages),
        )
        return result

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
    )
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a text generation request (non-chat format).

        Args:
            prompt: Text prompt
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Generated text response
        """
        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
        )

        logger.debug("Generated %d chars", len(response))
        return response

    def __repr__(self) -> str:
        return f"LLMClient(model='{self.model}')"
