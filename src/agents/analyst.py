"""
analyst.py — Synthesis agent for generating cited answers.

Receives retrieved passages and the original query, then uses an LLM to
produce a coherent, well-cited research answer.
"""

import logging
from typing import Optional

from src.utils.llm_client import LLMClient
from src.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────────

ANALYST_SYSTEM_PROMPT = """You are a research analyst AI. Your job is to synthesize information from retrieved ArXiv paper passages into a coherent, well-cited answer to a research question.

Rules:
1. ONLY use information from the provided passages — do not hallucinate facts
2. Cite papers using [Paper ID] format (e.g., [2301.12345])
3. When multiple papers discuss similar findings, synthesize across them
4. Organize your answer with clear structure (use headings if appropriate)
5. Explicitly note if the passages are insufficient to fully answer the question
6. Use precise technical language appropriate for a research audience
7. If passages contain conflicting findings, present both sides with citations

Format your response as a structured research answer."""


# ── Analyst Agent ────────────────────────────────────────────────────────────

class AnalystAgent:
    """Synthesizes retrieved passages into a coherent, cited answer."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        obs_logger: Optional[ObservabilityLogger] = None,
        max_context_passages: int = 15,
    ):
        """
        Args:
            llm_client: LLMClient instance (creates default if None)
            obs_logger: ObservabilityLogger instance (optional)
            max_context_passages: Maximum passages to include in context
        """
        self.llm = llm_client or LLMClient()
        self.obs_logger = obs_logger
        self.max_context_passages = max_context_passages

    def _format_context(self, passages: list[dict]) -> str:
        """Format retrieved passages into a context string for the LLM."""
        context_parts = []

        for i, passage in enumerate(passages[:self.max_context_passages]):
            metadata = passage.get("metadata", {})
            paper_id = metadata.get("paper_id", "unknown")
            title = metadata.get("title", "Untitled")
            year = metadata.get("year", "")
            chunk_type = metadata.get("chunk_type", "abstract")

            header = f"[Passage {i+1}] Paper: {paper_id} | Title: {title} | Year: {year} | Type: {chunk_type}"
            text = passage.get("text", "")

            context_parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def analyze(
        self,
        query: str,
        passages: list[dict],
        query_id: str = "",
    ) -> str:
        """
        Synthesize an answer from retrieved passages.

        Args:
            query: The original user research question
            passages: List of retrieved passage dicts
            query_id: Parent query ID for logging

        Returns:
            Synthesized answer string with citations
        """
        if not passages:
            no_result = (
                "I could not find sufficient information in the corpus to answer "
                "this question. The retrieval system returned no relevant passages. "
                "Consider rephrasing your query or checking that the relevant papers "
                "have been indexed."
            )
            return no_result

        # Format context
        context = self._format_context(passages)
        num_passages = min(len(passages), self.max_context_passages)

        # Build prompt
        user_message = (
            f"Research Question: {query}\n\n"
            f"I have retrieved {num_passages} relevant passages from ArXiv papers. "
            f"Please synthesize a comprehensive answer based on these passages.\n\n"
            f"Retrieved Passages:\n\n{context}"
        )

        # Call LLM
        answer = self.llm.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=ANALYST_SYSTEM_PROMPT,
            max_tokens=2048,
            temperature=0.3,
        )

        # Log decision
        if self.obs_logger and query_id:
            self.obs_logger.log_decision(
                query_id=query_id,
                agent_name="analyst",
                action="synthesize",
                input_summary=f"Query: {query[:200]} | {num_passages} passages",
                output_summary=f"Answer: {len(answer)} chars",
                reasoning=f"Used {num_passages}/{len(passages)} passages",
            )

        logger.info(
            "Analyst synthesized answer (%d chars) from %d passages",
            len(answer), num_passages,
        )

        return answer
