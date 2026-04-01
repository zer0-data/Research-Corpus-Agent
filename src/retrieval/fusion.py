"""
fusion.py — Reciprocal Rank Fusion for combining multiple retrieval results.

Implements 3-way RRF across dense (BGE), ColBERT, and sparse (BM25) retrievers.
Falls back to 2-way RRF when ColBERT is disabled (memory pressure on T4).

RRF formula: score(d) = Σ 1 / (k + rank_i(d))   where k=60
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_K = 60  # RRF constant (Cormack et al., 2009)


# ── HybridFuser ──────────────────────────────────────────────────────────────

class HybridFuser:
    """
    Reciprocal Rank Fusion across 2 or 3 retrieval result lists.

    Merges by doc_id, computes RRF score from each list's rank,
    and returns a deduplicated, re-ranked merged list.
    """

    def __init__(self, k: int = DEFAULT_K):
        """
        Args:
            k: RRF constant (default 60)
        """
        self.k = k

    def fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        colbert_results: list[dict] = None,
    ) -> list[dict]:
        """
        Perform Reciprocal Rank Fusion across retriever results.

        Args:
            dense_results: Ranked results from DenseRetriever
            sparse_results: Ranked results from SparseRetriever
            colbert_results: Ranked results from ColBERTRetriever (optional)

        Returns:
            Fused, deduplicated, re-ranked list of result dicts.
            Each dict has: doc_id, text, metadata, rrf_score, source_ranks
        """
        # Collect all result lists with labels
        retriever_lists = [
            ("dense", dense_results),
            ("sparse", sparse_results),
        ]
        if colbert_results is not None:
            retriever_lists.append(("colbert", colbert_results))

        # Accumulate RRF scores per doc_id
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_data: dict[str, dict] = {}
        source_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for retriever_name, ranked_list in retriever_lists:
            for rank, result in enumerate(ranked_list, start=1):
                doc_id = result["doc_id"]
                rrf_scores[doc_id] += 1.0 / (self.k + rank)
                source_ranks[doc_id][retriever_name] = rank

                # Keep the richest data (prefer entries with text)
                if doc_id not in doc_data or (not doc_data[doc_id].get("text") and result.get("text")):
                    doc_data[doc_id] = {
                        "doc_id": doc_id,
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                    }

        # Build fused output sorted by RRF score descending
        fused = []
        for doc_id in rrf_scores:
            entry = doc_data[doc_id].copy()
            entry["rrf_score"] = rrf_scores[doc_id]
            entry["source_ranks"] = dict(source_ranks[doc_id])
            fused.append(entry)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        logger.debug(
            "RRF fusion: %d unique docs from %d retrievers (top score: %.5f)",
            len(fused),
            len(retriever_lists),
            fused[0]["rrf_score"] if fused else 0.0,
        )

        return fused
