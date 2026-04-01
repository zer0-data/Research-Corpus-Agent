"""
retriever.py — Hybrid search agent for sub-task retrieval.

For each sub-query from the planner, calls HybridRetriever.search()
and deduplicates results across all sub-queries.
"""

import logging

from src.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


# ── Retriever Agent ──────────────────────────────────────────────────────────

class RetrieverAgent:
    """Runs hybrid search per sub-query and returns aggregated, deduplicated docs."""

    def __init__(self, hybrid_retriever: HybridRetriever = None):
        """
        Args:
            hybrid_retriever: HybridRetriever instance (created with defaults if None)
        """
        self.retriever = hybrid_retriever or HybridRetriever()

    def retrieve(self, sub_queries: list[str]) -> list[dict]:
        """
        Run hybrid search for each sub-query and deduplicate results.

        Args:
            sub_queries: List of sub-query strings from the planner

        Returns:
            List of unique doc dicts with metadata, deduplicated by doc_id
        """
        all_docs = []
        seen_ids = set()

        for i, sq in enumerate(sub_queries):
            logger.info(
                "Retrieving sub-query %d/%d: %s",
                i + 1, len(sub_queries), sq[:80],
            )

            results = self.retriever.search(query=sq, top_n=10)

            for doc in results:
                doc_id = doc.get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    doc["source_sub_query"] = sq
                    all_docs.append(doc)

        logger.info(
            "RetrieverAgent: %d unique docs from %d sub-queries",
            len(all_docs), len(sub_queries),
        )
        return all_docs
