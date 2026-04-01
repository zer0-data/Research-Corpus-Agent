"""
parsers.py — JSON parsing with retry logic for LLM outputs.

Handles common LLM output quirks: markdown fences, trailing text,
partial JSON, etc.
"""

import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── JSON Extraction Patterns ─────────────────────────────────────────────────

# Match JSON wrapped in markdown code fences
_FENCED_JSON = re.compile(
    r"```(?:json)?\s*\n?(.*?)```",
    re.DOTALL,
)

# Match a JSON array
_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)

# Match a JSON object
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


# ── Main Parser ──────────────────────────────────────────────────────────────

def parse_json_response(
    text: str,
    expect_list: bool = False,
    fallback: Optional[Any] = None,
) -> Any:
    """
    Extract and parse JSON from an LLM response.

    Tries multiple strategies in order:
    1. Direct json.loads on stripped text
    2. Extract from markdown code fences
    3. Regex extraction of JSON array or object
    4. Return fallback value

    Args:
        text: Raw LLM output text
        expect_list: If True, prioritize extracting a JSON array
        fallback: Value to return if all parsing fails (default None)

    Returns:
        Parsed JSON (dict or list), or fallback on failure
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to parse_json_response")
        return fallback

    cleaned = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown fences
    fence_match = _FENCED_JSON.search(cleaned)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Regex extraction
    if expect_list:
        array_match = _JSON_ARRAY.search(cleaned)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

    obj_match = _JSON_OBJECT.search(cleaned)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass

    # Try array as fallback if not already tried
    if not expect_list:
        array_match = _JSON_ARRAY.search(cleaned)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

    # Strategy 4: Fallback
    logger.warning(
        "Failed to parse JSON from LLM response (length=%d). "
        "First 200 chars: %s",
        len(text), text[:200],
    )
    return fallback


def extract_field(
    text: str,
    field: str,
    default: str = "",
) -> str:
    """
    Extract a specific field value from a JSON response.

    Args:
        text: Raw LLM output
        field: JSON field name to extract
        default: Default value if extraction fails

    Returns:
        Field value as string
    """
    parsed = parse_json_response(text)
    if isinstance(parsed, dict):
        return str(parsed.get(field, default))
    return default


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Safely serialize an object to JSON string."""
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        logger.error("JSON serialization failed: %s", e)
        return json.dumps({"error": str(e), "raw": str(obj)})
