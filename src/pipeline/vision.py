"""
vision.py — Figure and table extraction from PDF papers using Qwen2.5-VL.

Uses the HuggingFace Serverless Inference API with Qwen/Qwen2.5-VL-3B-Instruct
to describe figures and extract tables from academic PDFs. Returns embeddable
text chunks identical in format to regular text chunks.
"""

import io
import json
import logging
import os
import re
from pathlib import Path

import fitz  # pymupdf
from huggingface_hub import InferenceClient
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_FIGURES_PER_PAPER = 5
MIN_IMAGE_DIMENSION = 100  # Skip images smaller than 100x100 (icons/logos)
MAX_THUMBNAIL_SIZE = 512


# ── FigureExtractor ──────────────────────────────────────────────────────────

class FigureExtractor:
    """
    Extracts figures and tables from academic PDFs using Qwen2.5-VL
    via HuggingFace Serverless Inference API.
    """

    def __init__(self, hf_token: str = None):
        """
        Args:
            hf_token: HuggingFace API token. Falls back to HF_TOKEN env var.
        """
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable "
                "or pass hf_token= to FigureExtractor."
            )

        self.client = InferenceClient(
            model=VISION_MODEL_ID,
            token=token,
        )
        logger.info("FigureExtractor initialized with model: %s", VISION_MODEL_ID)

    # ── Figure Extraction ────────────────────────────────────────────────

    def extract_from_pdf(
        self,
        pdf_path: str,
        paper_id: str,
        title: str,
    ) -> list[dict]:
        """
        Extract and describe figures from a PDF using VL model.

        Opens the PDF, extracts embedded images, filters out small icons,
        and uses Qwen2.5-VL to generate text descriptions for up to
        MAX_FIGURES_PER_PAPER qualifying images.

        Args:
            pdf_path: Path to the PDF file
            paper_id: ArXiv paper ID
            title: Paper title

        Returns:
            List of chunk dicts with chunk_type="figure"
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        figure_chunks = []
        figure_index = 0

        for page_num in range(len(doc)):
            if figure_index >= MAX_FIGURES_PER_PAPER:
                break

            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_info in image_list:
                if figure_index >= MAX_FIGURES_PER_PAPER:
                    break

                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    image_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    # Skip small images (icons, logos, decorations)
                    if img.width < MIN_IMAGE_DIMENSION or img.height < MIN_IMAGE_DIMENSION:
                        continue

                    # Resize to thumbnail for API
                    img.thumbnail(
                        (MAX_THUMBNAIL_SIZE, MAX_THUMBNAIL_SIZE),
                        Image.LANCZOS,
                    )

                    # Call HF Inference API
                    response = self.client.visual_question_answering(
                        image=img,
                        question=(
                            "Describe this academic figure: chart type, "
                            "what is measured, main result. Be concise and technical."
                        ),
                    )

                    # Extract generated text from response
                    if isinstance(response, list) and len(response) > 0:
                        description = response[0].get("answer", "").strip()
                    elif hasattr(response, "generated_text"):
                        description = response.generated_text.strip()
                    elif isinstance(response, str):
                        description = response.strip()
                    else:
                        description = str(response).strip()

                    if not description:
                        continue

                    chunk = {
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_type": "figure",
                        "page_number": page_num + 1,
                        "figure_index": figure_index,
                        "text": description,
                        "metadata": {
                            "paper_id": paper_id,
                            "title": title,
                            "chunk_type": "figure",
                            "page_number": page_num + 1,
                        },
                    }
                    figure_chunks.append(chunk)
                    figure_index += 1

                    logger.debug(
                        "Figure %d extracted from page %d of %s",
                        figure_index, page_num + 1, paper_id,
                    )

                except Exception as e:
                    logger.warning(
                        "Failed to extract image xref=%d from page %d of %s: %s",
                        xref, page_num + 1, paper_id, e,
                    )
                    continue

        doc.close()

        logger.info(
            "Extracted %d figure chunks from %s (%s)",
            len(figure_chunks), paper_id, pdf_path,
        )
        return figure_chunks

    # ── Table Extraction ─────────────────────────────────────────────────

    def extract_tables_as_markdown(
        self,
        pdf_path: str,
        paper_id: str,
        title: str,
    ) -> list[dict]:
        """
        Extract tabular content from a PDF and convert to markdown format.

        Uses fitz to find text blocks with tabular structure (multiple
        tab/space-separated columns), then calls Qwen2.5-VL to produce
        clean markdown tables.

        Args:
            pdf_path: Path to the PDF file
            paper_id: ArXiv paper ID
            title: Paper title

        Returns:
            List of chunk dicts with chunk_type="table"
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        table_chunks = []
        table_index = 0

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Try pymupdf's built-in table finder first
            try:
                tables = page.find_tables()
                for table in tables:
                    table_data = table.extract()
                    if not table_data or len(table_data) < 2:
                        continue

                    # Convert raw table data to a text block for VL processing
                    raw_text = "\n".join(
                        "\t".join(str(cell) if cell else "" for cell in row)
                        for row in table_data
                    )

                    # Render the page region as an image for VL
                    rect = table.bbox
                    clip = fitz.Rect(rect)
                    pix = page.get_pixmap(clip=clip, dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    img.thumbnail(
                        (MAX_THUMBNAIL_SIZE, MAX_THUMBNAIL_SIZE),
                        Image.LANCZOS,
                    )

                    # Call VL model to produce clean markdown
                    response = self.client.visual_question_answering(
                        image=img,
                        question=(
                            "Convert this to a clean markdown table. "
                            "If not a table, return the text as-is. Plain text only."
                        ),
                    )

                    if isinstance(response, list) and len(response) > 0:
                        md_text = response[0].get("answer", "").strip()
                    elif hasattr(response, "generated_text"):
                        md_text = response.generated_text.strip()
                    elif isinstance(response, str):
                        md_text = response.strip()
                    else:
                        md_text = raw_text  # Fallback to raw extraction

                    if not md_text:
                        md_text = raw_text

                    chunk = {
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_type": "table",
                        "page_number": page_num + 1,
                        "figure_index": table_index,
                        "text": md_text,
                        "metadata": {
                            "paper_id": paper_id,
                            "title": title,
                            "chunk_type": "table",
                            "page_number": page_num + 1,
                        },
                    }
                    table_chunks.append(chunk)
                    table_index += 1

            except Exception as e:
                logger.warning(
                    "Table extraction failed on page %d of %s: %s",
                    page_num + 1, paper_id, e,
                )
                continue

            # Fallback: scan text blocks for tabular patterns
            if not table_chunks:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4] if len(block) > 4 else ""
                    if not isinstance(text, str):
                        continue

                    lines = text.strip().split("\n")
                    if len(lines) < 3:
                        continue

                    # Heuristic: a block is tabular if most lines have
                    # multiple tab or multi-space separated columns
                    tabular_lines = sum(
                        1 for line in lines
                        if len(re.split(r"\t|  {2,}", line.strip())) >= 3
                    )

                    if tabular_lines / len(lines) < 0.5:
                        continue

                    chunk = {
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_type": "table",
                        "page_number": page_num + 1,
                        "figure_index": table_index,
                        "text": text.strip(),
                        "metadata": {
                            "paper_id": paper_id,
                            "title": title,
                            "chunk_type": "table",
                            "page_number": page_num + 1,
                        },
                    }
                    table_chunks.append(chunk)
                    table_index += 1

        doc.close()

        logger.info(
            "Extracted %d table chunks from %s (%s)",
            len(table_chunks), paper_id, pdf_path,
        )
        return table_chunks


# ── Convenience Function ─────────────────────────────────────────────────────

def extract_all_visuals(
    pdf_path: str,
    paper_id: str,
    title: str,
    hf_token: str = None,
) -> list[dict]:
    """
    Extract all figures and tables from a PDF.

    Convenience wrapper that creates a FigureExtractor and runs both
    extraction methods.

    Args:
        pdf_path: Path to the PDF file
        paper_id: ArXiv paper ID
        title: Paper title
        hf_token: Optional HF token override

    Returns:
        Combined list of figure and table chunk dicts
    """
    extractor = FigureExtractor(hf_token=hf_token)
    figures = extractor.extract_from_pdf(pdf_path, paper_id, title)
    tables = extractor.extract_tables_as_markdown(pdf_path, paper_id, title)
    return figures + tables


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.vision <pdf_path> [paper_id] [title]")
        sys.exit(1)

    pdf = sys.argv[1]
    pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    ttl = sys.argv[3] if len(sys.argv) > 3 else ""

    results = extract_all_visuals(pdf, paper_id=pid, title=ttl)
    print(json.dumps(results, indent=2))
    print(f"\nTotal: {len(results)} chunks extracted")
