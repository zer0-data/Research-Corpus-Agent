"""
judge.py — LLM-as-judge for evaluating answer quality.

Scores generated answers on relevance, completeness, and citation accuracy
using a separate LLM call.
"""

import logging
from typing import Optional

from src.utils.llm_client import LLMClient
from src.utils.parsers import parse_json_response

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert research evaluation judge. Score the following answer to a research question on three dimensions, each on a 1-5 scale:

**Relevance** (1-5): Does the answer directly address the question?
  1 = Completely irrelevant
  2 = Tangentially related
  3 = Partially relevant
  4 = Mostly relevant
  5 = Directly and fully addresses the question

**Completeness** (1-5): Are all aspects of the question covered?
  1 = Major aspects missing
  2 = Several important aspects missing
  3 = Covers main points but misses some details
  4 = Covers most aspects well
  5 = Comprehensive coverage

**Citation Accuracy** (1-5): Are citations used correctly and verifiably?
  1 = No citations or all fabricated
  2 = Few citations, mostly unverifiable
  3 = Some citations present but inconsistent
  4 = Good citation coverage with minor issues
  5 = Excellent citations, all verifiable

Respond ONLY with a JSON object:
{
  "relevance": <score>,
  "completeness": <score>,
  "citation_accuracy": <score>,
  "overall": <average of three scores>,
  "justification": "<brief explanation of scores>"
}"""


# ── Judge ────────────────────────────────────────────────────────────────────

class LLMJudge:
    """LLM-based judge for evaluating answer quality."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            llm_client: LLMClient instance (creates default if None)
            model: Override model for judging (e.g., a stronger model)
        """
        if llm_client:
            self.llm = llm_client
        elif model:
            self.llm = LLMClient(model=model)
        else:
            self.llm = LLMClient()

    def judge(
        self,
        query: str,
        answer: str,
        reference_passages: list[dict] = None,
    ) -> dict:
        """
        Score an answer using the LLM judge.

        Args:
            query: The research question
            answer: The generated answer to evaluate
            reference_passages: Optional list of passages for context

        Returns:
            Dict with relevance, completeness, citation_accuracy,
            overall, and justification
        """
        # Build context
        context_str = ""
        if reference_passages:
            context_parts = []
            for i, p in enumerate(reference_passages[:10]):
                metadata = p.get("metadata", {})
                paper_id = metadata.get("paper_id", "unknown")
                context_parts.append(f"[{paper_id}] {p.get('text', '')[:300]}")
            context_str = f"\n\nAvailable source passages:\n" + "\n---\n".join(context_parts)

        user_message = (
            f"## Question\n{query}\n\n"
            f"## Answer to Evaluate\n{answer}"
            f"{context_str}"
        )

        response = self.llm.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=512,
        )

        # Parse scores
        default = {
            "relevance": 3,
            "completeness": 3,
            "citation_accuracy": 3,
            "overall": 3.0,
            "justification": "Could not parse judge response",
        }

        result = parse_json_response(response, fallback=default)
        if not isinstance(result, dict):
            result = default

        # Validate and compute overall
        for key in ["relevance", "completeness", "citation_accuracy"]:
            val = result.get(key, 3)
            result[key] = max(1, min(5, int(val) if isinstance(val, (int, float)) else 3))

        result["overall"] = round(
            (result["relevance"] + result["completeness"] + result["citation_accuracy"]) / 3, 2
        )

        logger.info(
            "Judge scores — R: %d, C: %d, CA: %d, Overall: %.2f",
            result["relevance"], result["completeness"],
            result["citation_accuracy"], result["overall"],
        )

        return result

    def batch_judge(
        self,
        items: list[dict],
    ) -> list[dict]:
        """
        Evaluate multiple query-answer pairs.

        Args:
            items: List of dicts with 'query', 'answer', and optional 'passages'

        Returns:
            List of judge result dicts
        """
        results = []
        for item in items:
            result = self.judge(
                query=item["query"],
                answer=item["answer"],
                reference_passages=item.get("passages"),
            )
            results.append(result)
        return results
