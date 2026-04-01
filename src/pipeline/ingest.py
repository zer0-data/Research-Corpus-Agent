"""
ingest.py — Download and filter ArXiv metadata from Kaggle.

Filters to CS/AI/ML papers (cs.AI, cs.LG, cs.CL) and saves as JSONL.
Target: ≥100K papers post-filtering.
"""

import json
import os
import sys
import subprocess
import zipfile
import logging
from pathlib import Path
from datetime import datetime, timezone

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

KAGGLE_DATASET = "Cornell-University/arxiv"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FILTERED_OUTPUT = DATA_DIR / "arxiv_filtered.jsonl"
SUMMARY_FILE = DATA_DIR / "ingestion_summary.json"

TARGET_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_kaggle_credentials() -> bool:
    """Verify that Kaggle API credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Also check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def _download_dataset(force: bool = False) -> Path:
    """
    Download the ArXiv dataset from Kaggle using the kaggle CLI.
    Returns the path to the extracted JSON file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    expected_file = RAW_DIR / "arxiv-metadata-oai-snapshot.json"
    if expected_file.exists() and not force:
        logger.info("Dataset already downloaded at %s", expected_file)
        return expected_file

    if not _check_kaggle_credentials():
        raise RuntimeError(
            "Kaggle API credentials not found.\n"
            "Option 1: Place kaggle.json at ~/.kaggle/kaggle.json\n"
            "Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
            "See: https://www.kaggle.com/docs/api#authentication"
        )

    logger.info("Downloading ArXiv dataset from Kaggle...")
    subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            KAGGLE_DATASET,
            "--path", str(RAW_DIR),
            "--unzip",
        ],
        check=True,
    )

    # If downloaded as zip (some versions of kaggle CLI don't auto-unzip)
    zip_path = RAW_DIR / "arxiv.zip"
    if zip_path.exists() and not expected_file.exists():
        logger.info("Extracting zip archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    if not expected_file.exists():
        # Try to find whatever JSON file was extracted
        json_files = list(RAW_DIR.glob("*.json"))
        if json_files:
            expected_file = json_files[0]
            logger.info("Found extracted file: %s", expected_file)
        else:
            raise FileNotFoundError(
                f"Expected ArXiv JSON file not found in {RAW_DIR}. "
                "Check Kaggle download output."
            )

    return expected_file


def _has_target_category(categories_str: str) -> bool:
    """Check if a paper's categories contain any target CS categories."""
    cats = set(categories_str.strip().split())
    return bool(cats & TARGET_CATEGORIES)


def _extract_record(paper: dict) -> dict:
    """Extract and normalize a single paper record."""
    return {
        "id": paper.get("id", ""),
        "title": paper.get("title", "").replace("\n", " ").strip(),
        "abstract": paper.get("abstract", "").replace("\n", " ").strip(),
        "authors": paper.get("authors", ""),
        "categories": paper.get("categories", ""),
        "update_date": paper.get("update_date", ""),
    }


# ── Main Pipeline ────────────────────────────────────────────────────────────

def ingest(force_download: bool = False) -> dict:
    """
    Run the full ingestion pipeline:
    1. Download ArXiv metadata from Kaggle
    2. Filter to CS/AI/ML papers
    3. Save as JSONL
    4. Write summary

    Returns:
        dict with ingestion statistics
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    raw_file = _download_dataset(force=force_download)
    logger.info("Raw dataset at: %s", raw_file)

    # Step 2: Stream, filter, and write
    total_scanned = 0
    total_kept = 0

    logger.info("Filtering papers to categories: %s", TARGET_CATEGORIES)

    with open(raw_file, "r", encoding="utf-8") as fin, \
         open(FILTERED_OUTPUT, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Filtering ArXiv papers", unit=" papers"):
            total_scanned += 1
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            categories = paper.get("categories", "")
            if _has_target_category(categories):
                record = _extract_record(paper)
                fout.write(json.dumps(record) + "\n")
                total_kept += 1

    # Step 3: Write summary
    summary = {
        "total_scanned": total_scanned,
        "total_kept": total_kept,
        "target_categories": sorted(TARGET_CATEGORIES),
        "output_file": str(FILTERED_OUTPUT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_dataset": KAGGLE_DATASET,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Ingestion complete: %d / %d papers kept (%.1f%%)",
        total_kept, total_scanned,
        100 * total_kept / max(total_scanned, 1),
    )

    return summary


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    result = ingest(force_download="--force" in sys.argv)
    print(json.dumps(result, indent=2))
