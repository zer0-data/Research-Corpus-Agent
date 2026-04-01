"""
critic.py — Answer quality critic agent.

Checks the analyst's answer for hallucinations, gaps, and contradictions
by comparing against the retrieved documents. Returns a structured verdict.
"""

import logging

from src.utils.llm_client import call_llm
from src.utils.parsers import parse_json_response

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = (
    "You are a critic reviewing a research answer. Check for: "
    "hallucinations (claims not in the docs), gaps (query aspects not addressed), "
    "contradictions. Classify the failure type. "
    "Return JSON only: "
    '{"verdict": "pass or revise", '
    '"failure_type": "hallucination or retrieval_failure or reasoning_failure or gap or null", '
    '"issues": ["list of strings"], '
    '"evidence": ["list of strings"], '
    '"revised_answer": "string or null"}'
)

# ── Default Verdict ──────────────────────────────────────────────────────────

_DEFAULT_VERDICT = {
    "verdict": "pass",
    "failure_type": None,
    "issues": [],
    "evidence": [],
    "revised_answer": None,
}


# ── Critic Agent ─────────────────────────────────────────────────────────────

class CriticAgent:
    """Evaluates answer quality and checks for hallucinations/gaps."""

    def critique(
        self,
        query: str,
        answer: str,
        docs: list[dict],
    ) -> dict:
        """
        Evaluate an answer for quality, accuracy, and completeness.

        Args:
            query: Original research question
            answer: Generated answer to evaluate
            docs: Retrieved documents used for the answer

        Returns:
            Dict with keys: verdict, failure_type, issues, evidence, revised_answer
        """
        # Build source context for the critic
        source_parts = []
        for i, doc in enumerate(docs[:5]):
            metadata = doc.get("metadata", {})
            title = metadata.get("title", doc.get("title", "Untitled"))
            paper_id = metadata.get("paper_id", doc.get("doc_id", "unknown"))
            text = doc.get("text", "")
            source_parts.append(
                f"[Source {i+1}] Title: {title} (ID: {paper_id})\n{text}"
            )

        sources = "\n\n---\n\n".join(source_parts)

        prompt = (
            f"<|im_start|>system\n{CRITIC_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Query: {query}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Source Documents:\n{sources}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        raw = call_llm(prompt, max_tokens=512)

        try:
            result = parse_json_response(raw)
        except ValueError:
            logger.warning("Critic parse failed, defaulting to pass verdict")
            return _DEFAULT_VERDICT.copy()

        if not isinstance(result, dict):
            return _DEFAULT_VERDICT.copy()

        # Normalize and validate fields
        result.setdefault("verdict", "pass")
        result.setdefault("failure_type", None)
        result.setdefault("issues", [])
        result.setdefault("evidence", [])
        result.setdefault("revised_answer", None)

        # Ensure verdict is valid
        if result["verdict"] not in ("pass", "revise"):
            result["verdict"] = "pass"

        # Ensure failure_type is valid
        valid_failures = {
            "hallucination", "retrieval_failure",
            "reasoning_failure", "gap", None,
        }
        if result["failure_type"] not in valid_failures:
            result["failure_type"] = None

        logger.info(
            "Critic: verdict=%s, failure_type=%s, issues=%d",
            result["verdict"],
            result["failure_type"],
            len(result["issues"]),
        )

        return result
