"""
embed.py — Sentence encoding and vector store indexing.

Encodes document chunks using BAAI/bge-large-en-v1.5, then stores to:
  1. ChromaDB (dense vectors)
  2. BM25 index (sparse, pickled)
  3. ColBERT PLAID index (late interaction, via pylate)

Supports checkpointing every 10K docs for resume-on-restart.
"""

import json
import os
import pickle
import logging
from pathlib import Path
from datetime import datetime, timezone

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
CHUNKS_INPUT = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
COLBERT_DIR = DATA_DIR / "colbert_index"
SUMMARY_FILE = DATA_DIR / "ingestion_summary.json"
CHECKPOINT_FILE = DATA_DIR / "embed_checkpoint.json"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DOC_PREFIX = "Represent this passage for retrieval: "
BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 10_000

CHROMA_COLLECTION = "arxiv_papers"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_chunks(start_from: int = 0) -> tuple[list[dict], int]:
    """Load chunks from JSONL, optionally skipping already-processed ones."""
    chunks = []
    total = 0
    with open(CHUNKS_INPUT, "r", encoding="utf-8") as f:
        for line in f:
            if total < start_from:
                total += 1
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            total += 1
    return chunks, total


def _save_checkpoint(processed_count: int):
    """Save progress checkpoint for resume capability."""
    checkpoint = {
        "processed_count": processed_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def _load_checkpoint() -> int:
    """Load last checkpoint, returns number of docs already processed."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
            return data.get("processed_count", 0)
    return 0


def _detect_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Dense Embedding (Chroma) ────────────────────────────────────────────────

def build_chroma_index(
    chunks: list[dict],
    model: SentenceTransformer,
    start_offset: int = 0,
) -> int:
    """
    Encode chunks with BGE and store in ChromaDB.

    Args:
        chunks: list of chunk dicts with 'text' and metadata fields
        model: loaded SentenceTransformer
        start_offset: global offset for chunk IDs (for resume)

    Returns:
        Number of chunks indexed
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total_indexed = 0

    for batch_start in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="Embedding → Chroma",
        unit=" batches",
    ):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [DOC_PREFIX + c["text"] for c in batch]

        # Encode
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        )

        # Prepare metadata and IDs
        ids = [f"chunk_{start_offset + batch_start + i}" for i in range(len(batch))]
        metadatas = [
            {
                "paper_id": c.get("paper_id", ""),
                "title": c.get("title", "")[:500],  # Chroma metadata size limit
                "authors": c.get("authors", "")[:500],
                "year": c.get("year", ""),
                "chunk_index": c.get("chunk_index", 0),
                "chunk_type": c.get("chunk_type", "abstract"),
            }
            for c in batch
        ]
        documents = [c["text"] for c in batch]

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents,
        )
        total_indexed += len(batch)

        # Checkpoint every N docs
        global_count = start_offset + batch_start + len(batch)
        if global_count % CHECKPOINT_INTERVAL < BATCH_SIZE:
            _save_checkpoint(global_count)
            logger.info("Checkpoint saved at %d docs", global_count)

    return total_indexed


# ── Sparse Index (BM25) ─────────────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> int:
    """
    Build a BM25 index over raw chunk text and pickle it.

    Returns:
        Number of documents indexed
    """
    logger.info("Building BM25 index over %d chunks...", len(chunks))

    # Tokenize (simple whitespace + lowercase)
    tokenized_corpus = [
        c["text"].lower().split() for c in tqdm(chunks, desc="Tokenizing for BM25")
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # Save both the index and the chunk IDs for retrieval
    bm25_data = {
        "index": bm25,
        "chunk_ids": [c.get("paper_id", "") + f"_c{c.get('chunk_index', 0)}" for c in chunks],
        "documents": [c["text"] for c in chunks],
        "metadatas": [
            {
                "paper_id": c.get("paper_id", ""),
                "title": c.get("title", ""),
                "authors": c.get("authors", ""),
                "year": c.get("year", ""),
                "chunk_index": c.get("chunk_index", 0),
                "chunk_type": c.get("chunk_type", "abstract"),
            }
            for c in chunks
        ],
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    logger.info("BM25 index saved to %s", BM25_PATH)
    return len(chunks)


# ── ColBERT Index (pylate) ───────────────────────────────────────────────────

def build_colbert_index(chunks: list[dict]) -> int:
    """
    Build a ColBERT PLAID index using pylate.

    Returns:
        Number of documents indexed, or 0 if pylate is unavailable.
    """
    try:
        from pylate import indexes, models as pylate_models
    except ImportError:
        logger.warning(
            "pylate not installed or import failed. "
            "Skipping ColBERT index build. Install with: pip install pylate"
        )
        return 0

    COLBERT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building ColBERT PLAID index over %d chunks...", len(chunks))

    try:
        # Load ColBERT model
        colbert_model = pylate_models.ColBERT(
            "colbert-ir/colBERTv2.0",
            device=_detect_device(),
        )

        documents = [c["text"] for c in chunks]
        doc_ids = [
            c.get("paper_id", "") + f"_c{c.get('chunk_index', 0)}"
            for c in chunks
        ]

        # Build index with checkpointing
        index = indexes.PLAID(
            index_folder=str(COLBERT_DIR),
            model=colbert_model,
        )

        # Process in batches with checkpointing
        for batch_start in tqdm(
            range(0, len(documents), CHECKPOINT_INTERVAL),
            desc="ColBERT indexing",
        ):
            batch_end = min(batch_start + CHECKPOINT_INTERVAL, len(documents))
            batch_docs = documents[batch_start:batch_end]
            batch_ids = doc_ids[batch_start:batch_end]

            index.add_documents(
                documents=batch_docs,
                document_ids=batch_ids,
            )

            _save_checkpoint(batch_end)
            logger.info("ColBERT checkpoint at %d docs", batch_end)

        logger.info("ColBERT index saved to %s", COLBERT_DIR)
        return len(documents)

    except Exception as e:
        logger.error("ColBERT indexing failed: %s", e)
        return 0


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_embedding(resume: bool = True) -> dict:
    """
    Run the full embedding pipeline:
    1. Load chunks (with optional resume from checkpoint)
    2. Build Chroma dense index
    3. Build BM25 sparse index
    4. Build ColBERT PLAID index
    5. Update ingestion summary

    Returns:
        dict with embedding statistics
    """
    if not CHUNKS_INPUT.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_INPUT}. Run chunk.py first."
        )

    # Resume support
    start_from = _load_checkpoint() if resume else 0
    if start_from > 0:
        logger.info("Resuming from checkpoint: %d docs already processed", start_from)

    # Load chunks
    chunks, total_in_file = _load_chunks(start_from=start_from)
    logger.info("Loaded %d chunks to process (total in file: %d)", len(chunks), total_in_file)

    if not chunks:
        logger.info("No new chunks to process.")
        return {"status": "no_new_chunks"}

    # Load embedding model
    device = _detect_device()
    logger.info("Loading embedding model %s on %s...", EMBEDDING_MODEL, device)
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Build indexes
    chroma_count = build_chroma_index(chunks, model, start_offset=start_from)
    bm25_count = build_bm25_index(chunks)
    colbert_count = build_colbert_index(chunks)

    # Final checkpoint
    _save_checkpoint(start_from + len(chunks))

    # Update summary
    stats = {
        "total_chunks_embedded": start_from + len(chunks),
        "chroma_indexed": chroma_count,
        "bm25_indexed": bm25_count,
        "colbert_indexed": colbert_count,
        "embedding_model": EMBEDDING_MODEL,
        "device": device,
        "chroma_dir": str(CHROMA_DIR),
        "bm25_path": str(BM25_PATH),
        "colbert_dir": str(COLBERT_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Merge with existing summary if it exists
    summary = {}
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "r") as f:
            summary = json.load(f)
    summary["embedding"] = stats

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Embedding pipeline complete: %s", stats)
    return stats


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    import sys
    result = run_embedding(resume="--no-resume" not in sys.argv)
    print(json.dumps(result, indent=2))
