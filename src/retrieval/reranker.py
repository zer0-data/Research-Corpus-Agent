"""
reranker.py — Cross-encoder re-ranking of retrieved passages.

Uses cross-encoder/ms-marco-MiniLM-L-12-v2 to re-score query-document pairs
and return a re-ranked list.
"""

import logging
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_BATCH_SIZE = 32


# ── Reranker ─────────────────────────────────────────────────────────────────

class Reranker:
    """Cross-encoder reranker for refining retrieval results."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model ID
            device: Device for the model ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for scoring
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading reranker: %s on %s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        logger.info("Reranker loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Re-rank candidate passages using the cross-encoder.

        Args:
            query: The search query
            candidates: List of candidate dicts, each must have a 'text' key.
                        Other keys (id, metadata, etc.) are preserved.
            top_k: Number of top results to return (None = all re-ranked)

        Returns:
            Re-ranked list of candidate dicts, sorted by cross-encoder score
            descending. Each dict gets a 'rerank_score' key added.
        """
        if not candidates:
            return []

        # Build query-document pairs
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach scores to candidates
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            entry = candidate.copy()
            entry["rerank_score"] = float(score)
            scored_candidates.append(entry)

        # Sort by rerank score descending
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]

        logger.debug(
            "Reranked %d candidates → top score: %.4f, bottom score: %.4f",
            len(scored_candidates),
            scored_candidates[0]["rerank_score"] if scored_candidates else 0,
            scored_candidates[-1]["rerank_score"] if scored_candidates else 0,
        )

        return scored_candidates
