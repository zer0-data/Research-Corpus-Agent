"""
retriever.py — Hybrid search agent for sub-task retrieval.

For each sub-task from the planner, delegates to the HybridRetriever pipeline
(dense + sparse + ColBERT → RRF → reranker). Returns aggregated context passages.
"""

import logging
from typing import Optional

from src.retrieval import HybridRetriever
from src.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)


# ── Retriever Agent ──────────────────────────────────────────────────────────

class RetrieverAgent:
    """Runs hybrid search per sub-task and returns aggregated context passages."""

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        obs_logger: Optional[ObservabilityLogger] = None,
        top_n: int = 10,
    ):
        """
        Args:
            hybrid_retriever: HybridRetriever instance (created with defaults if None)
            obs_logger: ObservabilityLogger instance
            top_n: Number of results per sub-task after reranking
        """
        self.retriever = hybrid_retriever or HybridRetriever(obs_logger=obs_logger)
        self.obs_logger = obs_logger
        self.top_n = top_n

    def retrieve_for_subtask(
        self,
        sub_task: dict,
        query_id: str = "",
    ) -> list[dict]:
        """
        Retrieve relevant passages for a single sub-task.

        Args:
            sub_task: Dict with keys: sub_query, intent, expected_doc_type
            query_id: Parent query ID for logging

        Returns:
            List of passage dicts with text, metadata, and scores
        """
        sub_query = sub_task.get("sub_query", "")
        if not sub_query:
            return []

        # Run full hybrid pipeline (dense + sparse + ColBERT → RRF → reranker)
        results = self.retriever.search(query=sub_query, top_n=self.top_n)

        # Log retrieval results
        if self.obs_logger and query_id:
            self.obs_logger.log_retrieval(
                query_id=query_id,
                sub_query=sub_query,
                retriever_type="hybrid_reranked",
                doc_ids=[r["doc_id"] for r in results],
                scores=[r.get("rerank_score", r.get("rrf_score", 0.0)) for r in results],
            )

        return results

    def retrieve(
        self,
        sub_tasks: list[dict],
        query_id: str = "",
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Retrieve passages for all sub-tasks and aggregate.

        Args:
            sub_tasks: List of sub-task dicts from the planner
            query_id: Parent query ID for logging
            deduplicate: Remove duplicate passages across sub-tasks

        Returns:
            Aggregated list of passage dicts
        """
        all_passages = []
        seen_ids = set()

        for i, sub_task in enumerate(sub_tasks):
            logger.info(
                "Retrieving for sub-task %d/%d: %s",
                i + 1, len(sub_tasks),
                sub_task.get("sub_query", "")[:80],
            )

            passages = self.retrieve_for_subtask(sub_task, query_id=query_id)

            for passage in passages:
                pid = passage.get("doc_id", "")
                if deduplicate and pid in seen_ids:
                    continue
                seen_ids.add(pid)
                passage["source_subtask"] = i
                all_passages.append(passage)

        # Log aggregated decision
        if self.obs_logger and query_id:
            self.obs_logger.log_decision(
                query_id=query_id,
                agent_name="retriever",
                action="retrieve",
                input_summary=f"{len(sub_tasks)} sub-tasks",
                output_summary=f"{len(all_passages)} unique passages retrieved",
                reasoning=f"Deduplicated: {deduplicate}",
            )

        logger.info(
            "Total retrieved: %d unique passages from %d sub-tasks",
            len(all_passages), len(sub_tasks),
        )

        return all_passages
