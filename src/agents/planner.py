"""
planner.py — Query decomposition agent.

Breaks a complex user query into atomic sub-tasks, each with a
sub-query, intent classification, and expected document type.
"""

import logging
from typing import Optional

from src.utils.llm_client import LLMClient
from src.utils.parsers import parse_json_response
from src.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are a research query planner. Your job is to decompose complex research questions into focused sub-tasks that can each be answered by searching a corpus of ArXiv CS/AI/ML papers.

For each sub-task, provide:
- sub_query: A focused, searchable query string
- intent: One of "factual", "comparative", "methodological", "survey", "trend"
- expected_doc_type: One of "abstract", "body", "figure", "table"

Rules:
1. Each sub-query should be self-contained and specific
2. Aim for 2-5 sub-tasks (avoid over-decomposition)
3. Order sub-tasks logically (foundational → specific)
4. Use technical terminology that would match ArXiv paper language

Respond ONLY with a JSON array of sub-task objects. No other text."""


PLANNER_EXAMPLE = """Example input: "Compare transformer architectures for long-context language modeling and discuss their computational trade-offs"

Example output:
[
  {
    "sub_query": "transformer architectures for long context language modeling",
    "intent": "survey",
    "expected_doc_type": "abstract"
  },
  {
    "sub_query": "efficient attention mechanisms linear attention sparse attention",
    "intent": "comparative",
    "expected_doc_type": "abstract"
  },
  {
    "sub_query": "computational complexity memory requirements long context transformers",
    "intent": "methodological",
    "expected_doc_type": "body"
  }
]"""


# ── Planner Agent ────────────────────────────────────────────────────────────

class PlannerAgent:
    """Decomposes complex queries into retrieval-friendly sub-tasks."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        obs_logger: Optional[ObservabilityLogger] = None,
    ):
        """
        Args:
            llm_client: LLMClient instance (created with defaults if None)
            obs_logger: ObservabilityLogger instance (optional)
        """
        self.llm = llm_client or LLMClient()
        self.obs_logger = obs_logger

    def plan(self, query: str) -> tuple[str, list[dict]]:
        """
        Decompose a query into sub-tasks.

        Args:
            query: The user's research question

        Returns:
            Tuple of (query_id, sub_tasks) where sub_tasks is a list of dicts
            with keys: sub_query, intent, expected_doc_type
        """
        # Log the query
        query_id = ""
        if self.obs_logger:
            query_id = self.obs_logger.log_query(query)

        # Call LLM
        messages = [
            {"role": "user", "content": f"{PLANNER_EXAMPLE}\n\nNow decompose this query:\n\n{query}"},
        ]

        response = self.llm.chat(
            messages=messages,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.2,
        )

        # Parse response
        sub_tasks = parse_json_response(response, expect_list=True, fallback=[])

        if not sub_tasks:
            logger.warning("Planner returned no sub-tasks, falling back to original query")
            sub_tasks = [
                {
                    "sub_query": query,
                    "intent": "survey",
                    "expected_doc_type": "abstract",
                }
            ]

        # Validate and normalize
        valid_intents = {"factual", "comparative", "methodological", "survey", "trend"}
        valid_doc_types = {"abstract", "body", "figure", "table"}

        for task in sub_tasks:
            if task.get("intent") not in valid_intents:
                task["intent"] = "survey"
            if task.get("expected_doc_type") not in valid_doc_types:
                task["expected_doc_type"] = "abstract"

        # Log decision
        if self.obs_logger:
            self.obs_logger.log_decision(
                query_id=query_id,
                agent_name="planner",
                action="plan",
                input_summary=query[:500],
                output_summary=f"{len(sub_tasks)} sub-tasks generated",
                reasoning=str(sub_tasks),
            )
            # Update query with sub-tasks
            self.obs_logger.log_query(query, sub_tasks=sub_tasks)

        logger.info("Planned %d sub-tasks for query: %s", len(sub_tasks), query[:100])
        return query_id, sub_tasks
