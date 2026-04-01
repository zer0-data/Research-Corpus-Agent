"""
llm_client.py — HuggingFace Serverless Inference API wrapper.

Uses Qwen/Qwen2.5-3B-Instruct via InferenceClient with tenacity retry logic.
"""

import os
import logging

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ── Client Setup ─────────────────────────────────────────────────────────────

_token = os.environ.get("HF_TOKEN")
if not _token:
    logger.warning(
        "HF_TOKEN environment variable not set. "
        "LLM calls will fail. Set it with: export HF_TOKEN=your_token"
    )

client = InferenceClient(
    model="Qwen/Qwen2.5-3B-Instruct",
    token=_token,
)


# ── LLM Call ─────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """
    Call the LLM via HuggingFace Serverless Inference API.

    Args:
        prompt: The full prompt string (system + user message combined)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response, stripped of whitespace
    """
    response = client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.1,
        do_sample=True,
        stop_sequences=["<|im_end|>"],
    )
    return response.strip()
