"""
retriever.py — Hybrid search agent for sub-task retrieval.

For each sub-task from the planner, runs fusion retrieval (dense + sparse)
followed by cross-encoder reranking. Returns aggregated context passages.
"""

import logging
from typing import Optional

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.fusion import FusionRetriever
from src.retrieval.reranker import Reranker
from src.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)


# ── Retriever Agent ──────────────────────────────────────────────────────────

class RetrieverAgent:
    """Runs hybrid search per sub-task and returns aggregated context passages."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        reranker: Optional[Reranker] = None,
        obs_logger: Optional[ObservabilityLogger] = None,
        fusion_k: int = 60,
        retrieval_top_k: int = 50,
        rerank_top_k: int = 10,
    ):
        """
        Args:
            dense_retriever: DenseRetriever instance
            sparse_retriever: SparseRetriever instance
            reranker: Reranker instance
            obs_logger: ObservabilityLogger instance
            fusion_k: RRF constant
            retrieval_top_k: Number of results per retriever before fusion
            rerank_top_k: Number of results after reranking
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker
        self.obs_logger = obs_logger
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

        # Build fusion retriever
        self.fusion = FusionRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            rrf_k=fusion_k,
        )

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

        # Step 1: Fusion retrieval (dense + sparse)
        fused_results = self.fusion.search(
            query=sub_query,
            top_k=self.retrieval_top_k,
        )

        # Log fusion results
        if self.obs_logger and query_id:
            self.obs_logger.log_retrieval(
                query_id=query_id,
                sub_query=sub_query,
                retriever_type="fusion",
                doc_ids=[r["id"] for r in fused_results],
                scores=[r.get("rrf_score", 0.0) for r in fused_results],
            )

        # Step 2: Reranking
        if self.reranker and fused_results:
            reranked = self.reranker.rerank(
                query=sub_query,
                candidates=fused_results,
                top_k=self.rerank_top_k,
            )

            # Log reranked results
            if self.obs_logger and query_id:
                self.obs_logger.log_retrieval(
                    query_id=query_id,
                    sub_query=sub_query,
                    retriever_type="reranked",
                    doc_ids=[r["id"] for r in reranked],
                    scores=[r.get("rerank_score", 0.0) for r in reranked],
                )

            return reranked

        return fused_results[:self.rerank_top_k]

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
                pid = passage.get("id", "")
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
