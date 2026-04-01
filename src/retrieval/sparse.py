"""
sparse.py — BM25 sparse retrieval using rank_bm25.

Loads the pickled BM25 index built by embed.py and provides search functionality.
"""

import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BM25_PATH = Path("data/bm25_index.pkl")


# ── BM25 Retriever ───────────────────────────────────────────────────────────

class SparseRetriever:
    """Sparse retrieval using BM25 (rank_bm25)."""

    def __init__(self, index_path: str = str(BM25_PATH)):
        """
        Load the pickled BM25 index.

        Args:
            index_path: Path to the pickled BM25 data file
        """
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. Run embed.py first."
            )

        logger.info("Loading BM25 index from %s...", index_path)
        with open(index_path, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["index"]
        self.chunk_ids = bm25_data["chunk_ids"]
        self.documents = bm25_data["documents"]
        self.metadatas = bm25_data["metadatas"]

        logger.info("BM25 index loaded — %d documents", len(self.documents))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search the BM25 index for the most relevant chunks.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of result dicts with keys:
                id, text, score, metadata
        """
        # Tokenize query (matching the indexing tokenization)
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break  # No more relevant results

            results.append({
                "id": self.chunk_ids[idx],
                "text": self.documents[idx],
                "score": float(scores[idx]),
                "metadata": self.metadatas[idx],
            })

        return results

    def batch_search(self, queries: list[str], top_k: int = 20) -> list[list[dict]]:
        """Run multiple queries."""
        return [self.search(q, top_k) for q in queries]
