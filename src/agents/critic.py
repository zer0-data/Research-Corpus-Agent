"""
critic.py — Answer quality critic agent.

Checks the analyst's answer for gaps, unsupported claims, and hallucinations
by comparing it against the retrieved passages. Returns structured feedback.
"""

import logging
from typing import Optional

from src.utils.llm_client import LLMClient
from src.utils.parsers import parse_json_response
from src.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are a research answer critic. Your job is to evaluate answers generated from retrieved ArXiv paper passages for quality, accuracy, and completeness.

Evaluate the answer on these dimensions:
1. **Faithfulness**: Are all claims supported by the provided passages? Flag any unsupported statements.
2. **Completeness**: Does the answer address all aspects of the question? Identify any gaps.
3. **Citation accuracy**: Are citations used correctly? Are sources properly attributed?
4. **Coherence**: Is the answer well-organized and logically structured?
5. **Hallucination check**: Does the answer contain any fabricated facts, paper titles, or findings not in the passages?

Respond with a JSON object:
{
  "verdict": "pass" | "fail" | "partial",
  "confidence": 0.0 to 1.0,
  "issues": [
    {
      "type": "hallucination" | "unsupported_claim" | "missing_coverage" | "citation_error" | "coherence_issue",
      "description": "specific description of the issue",
      "severity": "low" | "medium" | "high"
    }
  ],
  "strengths": ["list of things done well"],
  "summary": "one-paragraph overall assessment"
}"""


# ── Critic Agent ─────────────────────────────────────────────────────────────

class CriticAgent:
    """Evaluates answer quality and checks for hallucinations."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        obs_logger: Optional[ObservabilityLogger] = None,
    ):
        """
        Args:
            llm_client: LLMClient instance (creates default if None)
            obs_logger: ObservabilityLogger instance (optional)
        """
        self.llm = llm_client or LLMClient()
        self.obs_logger = obs_logger

    def _format_passages_for_review(self, passages: list[dict]) -> str:
        """Format passages for the critic's review."""
        parts = []
        for i, p in enumerate(passages[:15]):
            metadata = p.get("metadata", {})
            paper_id = metadata.get("paper_id", "unknown")
            title = metadata.get("title", "Untitled")
            parts.append(f"[Source {i+1}] ID: {paper_id} | Title: {title}\n{p.get('text', '')}")
        return "\n\n---\n\n".join(parts)

    def critique(
        self,
        query: str,
        answer: str,
        passages: list[dict],
        query_id: str = "",
    ) -> dict:
        """
        Evaluate an answer for quality, accuracy, and completeness.

        Args:
            query: Original research question
            answer: Generated answer to evaluate
            passages: Retrieved passages used to generate the answer
            query_id: Parent query ID for logging

        Returns:
            Dict with verdict, confidence, issues, strengths, summary
        """
        sources = self._format_passages_for_review(passages)

        user_message = (
            f"## Research Question\n{query}\n\n"
            f"## Generated Answer\n{answer}\n\n"
            f"## Source Passages\n{sources}\n\n"
            f"Please evaluate the answer quality. Respond with the JSON evaluation."
        )

        response = self.llm.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=CRITIC_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=1024,
        )

        # Parse response
        default_result = {
            "verdict": "partial",
            "confidence": 0.5,
            "issues": [{"type": "coherence_issue", "description": "Could not parse critic response", "severity": "medium"}],
            "strengths": [],
            "summary": "Critic evaluation could not be fully parsed.",
        }

        result = parse_json_response(response, fallback=default_result)
        if not isinstance(result, dict):
            result = default_result

        # Ensure required fields
        result.setdefault("verdict", "partial")
        result.setdefault("confidence", 0.5)
        result.setdefault("issues", [])
        result.setdefault("strengths", [])
        result.setdefault("summary", "")

        # Log verdict
        if self.obs_logger and query_id:
            issue_descriptions = [
                iss.get("description", "") if isinstance(iss, dict) else str(iss)
                for iss in result["issues"]
            ]
            self.obs_logger.log_verdict(
                query_id=query_id,
                verdict=result["verdict"],
                confidence=result["confidence"],
                issues=issue_descriptions,
                answer_excerpt=answer[:500],
            )

            self.obs_logger.log_decision(
                query_id=query_id,
                agent_name="critic",
                action="critique",
                input_summary=f"Query: {query[:200]} | Answer: {len(answer)} chars",
                output_summary=f"Verdict: {result['verdict']} ({result['confidence']:.2f})",
                reasoning=result.get("summary", ""),
            )

        logger.info(
            "Critic verdict: %s (confidence: %.2f, issues: %d)",
            result["verdict"], result["confidence"], len(result["issues"]),
        )

        return result
