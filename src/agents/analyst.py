"""
analyst.py — Synthesis agent for generating cited answers.

Receives the original query and top retrieved documents, then uses Qwen2.5
to produce a technical, cited research answer.
"""

import logging

from src.utils.llm_client import call_llm

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────────

ANALYST_SYSTEM_PROMPT = (
    "You are a research analyst. Answer the query using only the provided papers. "
    "Cite paper titles inline. Be specific and technical."
)


# ── Analyst Agent ────────────────────────────────────────────────────────────

class AnalystAgent:
    """Synthesizes retrieved passages into a coherent, cited answer."""

    def analyze(self, query: str, docs: list[dict]) -> str:
        """
        Synthesize an answer from the query and retrieved documents.

        Uses top 5 documents to build the context window.

        Args:
            query: The original user research question
            docs: List of retrieved doc dicts (must have 'text' and 'metadata')

        Returns:
            Synthesized answer string with inline citations
        """
        if not docs:
            return (
                "No relevant documents were found in the corpus. "
                "Consider rephrasing the query or verifying that the "
                "relevant papers have been indexed."
            )

        # Build context from top 5 docs
        context_parts = []
        for i, doc in enumerate(docs[:5]):
            metadata = doc.get("metadata", {})
            title = metadata.get("title", doc.get("title", "Untitled"))
            paper_id = metadata.get("paper_id", doc.get("doc_id", "unknown"))
            text = doc.get("text", "")

            context_parts.append(
                f"[Paper {i+1}] Title: {title} (ID: {paper_id})\n{text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"<|im_start|>system\n{ANALYST_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Query: {query}\n\n"
            f"Retrieved Papers:\n\n{context}\n\n"
            f"Provide a comprehensive, cited answer.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        answer = call_llm(prompt, max_tokens=512)

        logger.info(
            "Analyst: generated %d-char answer from %d docs for: %s",
            len(answer), min(len(docs), 5), query[:80],
        )
        return answer
