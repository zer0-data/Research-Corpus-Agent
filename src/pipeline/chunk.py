"""
chunk.py — Chunking logic for ArXiv paper data.

Strategies:
- Abstract chunk: title + abstract combined as a single chunk per paper
- Body chunks: paragraph-level splits (~3-5 sentences, max 512 tokens)
- Figure/table chunks are produced by vision.py and merged here if present

Output: data/chunks.jsonl
"""

import json
import re
import logging
from pathlib import Path
from typing import Generator

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
FILTERED_INPUT = DATA_DIR / "arxiv_filtered.jsonl"
CHUNKS_OUTPUT = DATA_DIR / "chunks.jsonl"

MAX_CHUNK_TOKENS = 512
TARGET_SENTENCES_PER_CHUNK = (3, 5)  # min, max sentences

# Simple sentence boundary regex (handles ., !, ? followed by space/newline)
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _approx_token_count(text: str) -> int:
    """Rough token count: split on whitespace (~1.3 tokens per word on average)."""
    return int(len(text.split()) * 1.3)


def _extract_year(update_date: str) -> str:
    """Extract year from update_date string (format: YYYY-MM-DD or similar)."""
    if update_date and len(update_date) >= 4:
        return update_date[:4]
    return "unknown"


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_body(text: str, min_sent: int = 3, max_sent: int = 5) -> list[str]:
    """
    Split body text into paragraph-level chunks.
    Each chunk has 3-5 sentences and ≤512 tokens.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _approx_token_count(sentence)

        # If adding this sentence would exceed limits, flush current chunk
        if current_chunk and (
            len(current_chunk) >= max_sent
            or current_tokens + sent_tokens > MAX_CHUNK_TOKENS
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    # Flush remaining
    if current_chunk:
        # If too short, merge with previous chunk if possible
        if len(current_chunk) < min_sent and chunks:
            chunks[-1] += " " + " ".join(current_chunk)
        else:
            chunks.append(" ".join(current_chunk))

    return chunks


# ── Main Chunking ────────────────────────────────────────────────────────────

def chunk_paper(paper: dict) -> Generator[dict, None, None]:
    """
    Generate chunks from a single paper record.

    Yields chunk dicts with:
        paper_id, title, authors, year, chunk_index, chunk_type, text
    """
    paper_id = paper.get("id", "")
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    authors = paper.get("authors", "")
    year = _extract_year(paper.get("update_date", ""))

    chunk_index = 0

    # Abstract chunk: title + abstract
    if abstract:
        abstract_text = f"{title}. {abstract}" if title else abstract
        yield {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "chunk_index": chunk_index,
            "chunk_type": "abstract",
            "text": abstract_text,
        }
        chunk_index += 1

    # Body chunks (if full text is available)
    full_text = paper.get("full_text", "")
    if full_text and full_text.strip():
        body_chunks = _chunk_body(full_text)
        for body_text in body_chunks:
            yield {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "chunk_index": chunk_index,
                "chunk_type": "body",
                "text": body_text,
            }
            chunk_index += 1


def run_chunking() -> dict:
    """
    Run the full chunking pipeline.

    Reads data/arxiv_filtered.jsonl, produces data/chunks.jsonl.
    Returns summary statistics.
    """
    if not FILTERED_INPUT.exists():
        raise FileNotFoundError(
            f"Filtered input not found at {FILTERED_INPUT}. "
            "Run ingest.py first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_papers = 0
    total_chunks = 0
    chunks_by_type = {"abstract": 0, "body": 0, "figure": 0, "table": 0}

    with open(FILTERED_INPUT, "r", encoding="utf-8") as fin, \
         open(CHUNKS_OUTPUT, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Chunking papers", unit=" papers"):
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_papers += 1

            for chunk in chunk_paper(paper):
                fout.write(json.dumps(chunk) + "\n")
                total_chunks += 1
                chunk_type = chunk.get("chunk_type", "abstract")
                chunks_by_type[chunk_type] = chunks_by_type.get(chunk_type, 0) + 1

    summary = {
        "total_papers": total_papers,
        "total_chunks": total_chunks,
        "chunks_by_type": chunks_by_type,
        "output_file": str(CHUNKS_OUTPUT),
    }

    logger.info(
        "Chunking complete: %d papers → %d chunks  %s",
        total_papers, total_chunks, chunks_by_type,
    )

    return summary


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    result = run_chunking()
    print(json.dumps(result, indent=2))
