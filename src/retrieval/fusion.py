"""
fusion.py — Reciprocal Rank Fusion (RRF) for combining multiple retrieval results.

Merges ranked lists from dense, sparse, and optionally ColBERT retrievers
using the RRF formula: score(d) = Σ 1 / (k + rank_i(d))
"""

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_K = 60  # RRF constant (standard value from the original paper)


# ── RRF Implementation ───────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = DEFAULT_K,
    top_k: Optional[int] = None,
) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Formula: score(d) = Σ 1 / (k + rank_i(d))

    Args:
        ranked_lists: List of ranked result lists. Each result dict must
                      have at least an 'id' key. Other keys (text, metadata)
                      are preserved from the first occurrence.
        k: RRF constant (default 60, from Cormack et al. 2009)
        top_k: Number of results to return (None = all)

    Returns:
        Fused ranked list of result dicts, sorted by RRF score descending.
        Each dict has 'id', 'text', 'metadata', 'rrf_score',
        and 'source_ranks' showing per-retriever ranks.
    """
    # Accumulate RRF scores
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_data: dict[str, dict] = {}
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for list_idx, ranked_list in enumerate(ranked_lists):
        retriever_name = f"retriever_{list_idx}"

        for rank, result in enumerate(ranked_list, start=1):
            doc_id = result["id"]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            source_ranks[doc_id][retriever_name] = rank

            # Keep first occurrence's data
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "id": doc_id,
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                }

    # Build fused results
    fused = []
    for doc_id, score in rrf_scores.items():
        entry = doc_data[doc_id].copy()
        entry["rrf_score"] = score
        entry["source_ranks"] = dict(source_ranks[doc_id])
        fused.append(entry)

    # Sort by RRF score descending
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    if top_k is not None:
        fused = fused[:top_k]

    return fused


# ── Fusion Retriever (Convenience Wrapper) ───────────────────────────────────

class FusionRetriever:
    """
    Combines dense, sparse, and optionally ColBERT retrievers using RRF.
    """

    def __init__(
        self,
        dense_retriever=None,
        sparse_retriever=None,
        colbert_retriever=None,
        rrf_k: int = DEFAULT_K,
    ):
        """
        Args:
            dense_retriever: DenseRetriever instance (or None to skip)
            sparse_retriever: SparseRetriever instance (or None to skip)
            colbert_retriever: ColBERT retriever instance (or None to skip)
            rrf_k: RRF constant
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.colbert = colbert_retriever
        self.rrf_k = rrf_k

        active = sum(1 for r in [self.dense, self.sparse, self.colbert] if r is not None)
        logger.info("FusionRetriever initialized with %d active retrievers", active)

    def search(self, query: str, top_k: int = 20, per_retriever_k: int = 50) -> list[dict]:
        """
        Run hybrid search across all active retrievers and fuse results.

        Args:
            query: Search query string
            top_k: Number of final fused results to return
            per_retriever_k: Number of results to fetch from each retriever

        Returns:
            Fused ranked list of result dicts
        """
        ranked_lists = []

        if self.dense is not None:
            try:
                dense_results = self.dense.search(query, top_k=per_retriever_k)
                ranked_lists.append(dense_results)
                logger.debug("Dense returned %d results", len(dense_results))
            except Exception as e:
                logger.error("Dense retrieval failed: %s", e)

        if self.sparse is not None:
            try:
                sparse_results = self.sparse.search(query, top_k=per_retriever_k)
                ranked_lists.append(sparse_results)
                logger.debug("Sparse returned %d results", len(sparse_results))
            except Exception as e:
                logger.error("Sparse retrieval failed: %s", e)

        if self.colbert is not None:
            try:
                colbert_results = self.colbert.search(query, top_k=per_retriever_k)
                ranked_lists.append(colbert_results)
                logger.debug("ColBERT returned %d results", len(colbert_results))
            except Exception as e:
                logger.error("ColBERT retrieval failed: %s", e)

        if not ranked_lists:
            logger.warning("No retriever returned results for query: %s", query[:100])
            return []

        return reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, top_k=top_k)
