"""
reranker.py — Cross-encoder re-ranking of retrieved passages.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 loaded locally on T4 GPU
to re-score (query, passage) pairs for precision reranking.
"""

import logging
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 64


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── CrossEncoderReranker ─────────────────────────────────────────────────────

class CrossEncoderReranker:
    """Cross-encoder reranker for refining retrieval results."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model ID
            device: Device (None = auto-detect)
            batch_size: Batch size for scoring
        """
        device = device or _detect_device()

        logger.info("Loading CrossEncoderReranker: %s on %s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        logger.info("CrossEncoderReranker ready")

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        """
        Re-rank candidate documents using cross-encoder scoring.

        Args:
            query: The search query
            docs: List of candidate dicts, each must have a 'text' key
            top_n: Number of top results to return

        Returns:
            Re-ranked top_n list sorted by cross-encoder score descending.
            Each dict gets a 'rerank_score' key added.
        """
        if not docs:
            return []

        # Build (query, passage) pairs
        pairs = [(query, doc["text"]) for doc in docs if doc.get("text")]

        # If some docs had no text, we can only score those with text
        scorable_docs = [doc for doc in docs if doc.get("text")]
        unscorable_docs = [doc for doc in docs if not doc.get("text")]

        if not pairs:
            logger.warning("No scorable documents (all missing text)")
            return docs[:top_n]

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach scores
        scored_docs = []
        for doc, score in zip(scorable_docs, scores):
            entry = doc.copy()
            entry["rerank_score"] = float(score)
            scored_docs.append(entry)

        # Sort by rerank score descending
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Append unscorable docs at the end (below the scored ones)
        for doc in unscorable_docs:
            entry = doc.copy()
            entry["rerank_score"] = float("-inf")
            scored_docs.append(entry)

        result = scored_docs[:top_n]

        logger.debug(
            "Reranked %d docs → top_n=%d (best: %.4f, worst: %.4f)",
            len(docs), len(result),
            result[0]["rerank_score"] if result else 0,
            result[-1]["rerank_score"] if result else 0,
        )

        return result
