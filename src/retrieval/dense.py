"""
dense.py — Dense and ColBERT retrieval for the ArXiv Research Corpus Agent.

DenseRetriever: BAAI/bge-large-en-v1.5 via SentenceTransformer + ChromaDB
ColBERTRetriever: colbert-ir/colbertv2.0 via pylate (late interaction)

All models run locally on T4 GPU. No HuggingFace Inference API.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

CHROMA_DIR = Path("data/chroma_db")
COLBERT_INDEX_DIR = Path("data/colbert_index")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHROMA_COLLECTION = "arxiv_papers"


def _detect_device() -> str:
    """Detect best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Dense Retriever (BGE + Chroma) ───────────────────────────────────────────

class DenseRetriever:
    """Dense retrieval using BGE-large-en-v1.5 embeddings + ChromaDB."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        chroma_dir: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: SentenceTransformer model ID
            chroma_dir: Path to persistent Chroma directory
            collection_name: Name of the Chroma collection
            device: Device for the model (None = auto-detect)
        """
        device = device or _detect_device()

        logger.info("Loading DenseRetriever: %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(name=collection_name)
        self.device = device

        logger.info(
            "DenseRetriever ready — collection has %d documents",
            self.collection.count(),
        )

    def retrieve(self, query: str, top_k: int = 50) -> list[dict]:
        """
        Retrieve the most relevant chunks via dense vector search.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of result dicts: {doc_id, text, metadata, score}
        """
        # Encode query with BGE prefix
        query_embedding = self.model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack results (Chroma returns lists of lists)
        output = []
        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "doc_id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
                })

        return output


# ── ColBERT Retriever (pylate) ───────────────────────────────────────────────

class ColBERTRetriever:
    """Late-interaction retrieval using ColBERT v2 via pylate."""

    def __init__(
        self,
        index_folder: str = str(COLBERT_INDEX_DIR),
        index_name: str = "arxiv",
    ):
        """
        Args:
            index_folder: Path to the ColBERT PLAID index directory
            index_name: Name of the pylate index
        """
        from pylate import models, indexes, retrieve as pylate_retrieve

        logger.info("Loading ColBERTRetriever from %s/%s", index_folder, index_name)

        self.model = models.ColBERT("colbert-ir/colbertv2.0")
        self.index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
        )
        self.retriever = pylate_retrieve.ColBERT(index=self.index)

        logger.info("ColBERTRetriever ready")

    def retrieve(self, query: str, top_k: int = 50) -> list[dict]:
        """
        Retrieve the most relevant chunks via ColBERT late interaction.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of result dicts: {doc_id, text, metadata, score}
        """
        # Encode query for ColBERT
        query_embeddings = self.model.encode([query], is_query=True)

        # Retrieve from PLAID index
        results = self.retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=top_k,
        )

        # results is a list (per query) of lists of (doc_id, score) tuples
        output = []
        if results and len(results) > 0:
            for doc_id, score in results[0]:
                output.append({
                    "doc_id": str(doc_id),
                    "text": "",  # ColBERT index doesn't store text; filled by fusion
                    "metadata": {},
                    "score": float(score),
                })

        return output
