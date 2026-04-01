"""
sparse.py — BM25 sparse retrieval using rank_bm25.

Loads the pickled BM25 index built by embed.py and provides ranked retrieval.
All computation is local — no external API calls.
"""

import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BM25_PATH = Path("data/bm25_index.pkl")


# ── Sparse Retriever ─────────────────────────────────────────────────────────

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

        logger.info("SparseRetriever ready — %d documents indexed", len(self.documents))

    def retrieve(self, query: str, top_k: int = 50) -> list[dict]:
        """
        Retrieve the most relevant chunks via BM25 keyword scoring.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of result dicts: {doc_id, text, metadata, score}
        """
        # Tokenize query (matching the indexing tokenization: whitespace + lowercase)
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
                "doc_id": self.chunk_ids[idx],
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(scores[idx]),
            })

        return results
