"""
planner.py — Query decomposition agent.

Breaks a complex user query into at most 3 focused sub-queries
for searching the ArXiv paper database.
"""

import logging

from src.utils.llm_client import call_llm
from src.utils.parsers import parse_json_response

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = (
    "You are a research query planner. Break the user query into at most 3 "
    "specific sub-queries for searching an academic paper database. "
    'Return a JSON array of strings only. Example: '
    '["sub-query 1", "sub-query 2", "sub-query 3"]'
)


# ── Planner Agent ────────────────────────────────────────────────────────────

class PlannerAgent:
    """Decomposes complex queries into retrieval-friendly sub-queries."""

    def plan(self, query: str) -> list[str]:
        """
        Decompose a query into at most 3 sub-queries.

        Args:
            query: The user's research question

        Returns:
            List of sub-query strings (max 3)
        """
        prompt = (
            f"<|im_start|>system\n{PLANNER_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        raw = call_llm(prompt, max_tokens=256)

        try:
            sub_queries = parse_json_response(raw)
        except ValueError:
            logger.warning("Planner parse failed, falling back to original query")
            sub_queries = [query]

        # Validate: must be a list of strings, max 3
        if not isinstance(sub_queries, list):
            sub_queries = [query]

        sub_queries = [
            str(sq) for sq in sub_queries if isinstance(sq, str) and sq.strip()
        ][:3]

        if not sub_queries:
            sub_queries = [query]

        logger.info("Planner: %d sub-queries for: %s", len(sub_queries), query[:80])
        return sub_queries
