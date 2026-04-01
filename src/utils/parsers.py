"""
parsers.py — JSON parsing with LLM-based retry logic.

Extracts JSON from LLM outputs, retrying via the LLM itself if parsing fails.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_json_response(raw: str) -> dict | list:
    """
    Parse JSON from an LLM response with up to 3 attempts.

    Strategy:
    1. Strip markdown code fences, attempt json.loads
    2. On failure, call the LLM to fix the JSON
    3. Repeat up to 3 times total

    Args:
        raw: Raw LLM output text

    Returns:
        Parsed JSON (dict or list)

    Raises:
        ValueError: If JSON parsing fails after 3 attempts
    """
    from src.utils.llm_client import call_llm

    for attempt in range(3):
        try:
            # Strip markdown fences and whitespace
            clean = re.sub(r"```json|```", "", raw).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            if attempt < 2:
                logger.warning(
                    "JSON parse failed (attempt %d/3), retrying via LLM",
                    attempt + 1,
                )
                raw = call_llm(
                    raw + "\nReturn valid JSON only. No explanation:",
                    max_tokens=512,
                )

    raise ValueError(
        f"Failed to parse JSON after 3 attempts. Last raw output: {raw[:200]}"
    )
