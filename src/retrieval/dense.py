"""
dense.py — Dense retrieval using BGE sentence encoder + ChromaDB vector search.

Wraps the BAAI/bge-large-en-v1.5 model for query encoding and queries the
Chroma persistent collection built by embed.py.
"""

import logging
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

CHROMA_DIR = Path("data/chroma_db")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this query for retrieval: "
CHROMA_COLLECTION = "arxiv_papers"


# ── Dense Retriever ──────────────────────────────────────────────────────────

class DenseRetriever:
    """Dense retrieval using BGE embeddings + ChromaDB."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        chroma_dir: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION,
        device: Optional[str] = None,
    ):
        """
        Initialize the dense retriever.

        Args:
            model_name: SentenceTransformer model ID
            chroma_dir: Path to persistent Chroma directory
            collection_name: Name of the Chroma collection
            device: Device for the model ('cuda', 'cpu', or None for auto)
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading dense retriever: %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(name=collection_name)
        self.device = device

        logger.info(
            "Dense retriever ready — collection has %d documents",
            self.collection.count(),
        )

    def encode_query(self, query: str) -> list[float]:
        """Encode a query string using BGE with query prefix."""
        embedding = self.model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search the Chroma collection for the most relevant chunks.

        Args:
            query: Search query string
            top_k: Number of results to return
            where: Optional Chroma metadata filter

        Returns:
            List of result dicts with keys:
                id, text, score, metadata
        """
        query_embedding = self.encode_query(query)

        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            search_kwargs["where"] = where

        results = self.collection.query(**search_kwargs)

        # Unpack Chroma results (they come as lists of lists)
        output = []
        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "score": 1 - results["distances"][0][i],  # cosine distance → similarity
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return output

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 20,
    ) -> list[list[dict]]:
        """Run multiple queries in batch."""
        embeddings = self.model.encode(
            [QUERY_PREFIX + q for q in queries],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        results = self.collection.query(
            query_embeddings=embeddings.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        all_outputs = []
        for q_idx in range(len(queries)):
            output = []
            if results and results["ids"]:
                for i in range(len(results["ids"][q_idx])):
                    output.append({
                        "id": results["ids"][q_idx][i],
                        "text": results["documents"][q_idx][i] if results["documents"] else "",
                        "score": 1 - results["distances"][q_idx][i],
                        "metadata": results["metadatas"][q_idx][i] if results["metadatas"] else {},
                    })
            all_outputs.append(output)

        return all_outputs
