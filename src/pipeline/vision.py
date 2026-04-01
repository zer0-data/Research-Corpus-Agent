"""
vision.py — Figure and table extraction from PDF papers using Qwen2.5-VL.

Renders PDF pages to images via PyMuPDF, then uses Qwen2.5-VL to detect
and describe figures/tables, producing structured chunk records.
"""

import io
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

VISION_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_DPI = 200
MAX_IMAGE_SIZE = 1280  # Max dimension for model input


# ── Model Loading ────────────────────────────────────────────────────────────

_model = None
_processor = None


def _load_vision_model(device: Optional[str] = None):
    """Load Qwen2.5-VL model and processor (lazy, singleton)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading vision model %s on %s...", VISION_MODEL_ID, device)

    _processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)

    load_kwargs = {"torch_dtype": torch.float16}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except ImportError:
            logger.warning("bitsandbytes not available; loading in float16 without quantization")

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VISION_MODEL_ID, **load_kwargs
    )

    if device != "cuda":
        _model = _model.to(device)

    _model.eval()
    logger.info("Vision model loaded successfully")
    return _model, _processor


# ── PDF Rendering ────────────────────────────────────────────────────────────

def _render_pdf_pages(pdf_path: str, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    """Render all pages of a PDF to PIL images using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at specified DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Resize if too large
        if max(img.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        images.append(img)

    doc.close()
    return images


# ── Vision Analysis ──────────────────────────────────────────────────────────

FIGURE_TABLE_PROMPT = """Analyze this page from a scientific paper. 
If there are any figures or tables present, describe each one in detail:
1. Identify whether it is a FIGURE or TABLE
2. Provide the figure/table number if visible
3. Give a detailed description of its content, data, or visualization
4. Explain what the figure/table demonstrates in the context of the paper

If there are NO figures or tables on this page, respond with exactly: NO_FIGURES_OR_TABLES

Format your response as JSON:
[
  {
    "type": "figure" or "table",
    "number": "Figure 1" or "Table 2" etc.,
    "description": "detailed description",
    "significance": "what it shows/demonstrates"
  }
]
"""


def _analyze_page(
    image: Image.Image,
    page_num: int,
    model,
    processor,
) -> list[dict]:
    """
    Use Qwen2.5-VL to detect and describe figures/tables on a page.

    Returns list of dicts with type, number, description, significance.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": FIGURE_TABLE_PROMPT},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False,
        )

    # Decode only the generated tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if "NO_FIGURES_OR_TABLES" in response:
        return []

    # Parse JSON response
    try:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            items = json.loads(json_match.group())
            for item in items:
                item["page_num"] = page_num
            return items
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse vision model output for page %d", page_num)

    return []


# ── Main Pipeline ────────────────────────────────────────────────────────────

def extract_figures_tables(
    pdf_path: str,
    paper_id: str = "",
    title: str = "",
    authors: str = "",
    year: str = "",
) -> list[dict]:
    """
    Extract figure and table descriptions from a PDF paper.

    Args:
        pdf_path: Path to the PDF file
        paper_id: ArXiv paper ID
        title: Paper title
        authors: Paper authors
        year: Publication year

    Returns:
        List of chunk dicts with chunk_type="figure" or "table"
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    model, processor = _load_vision_model()

    logger.info("Rendering PDF pages: %s", pdf_path)
    pages = _render_pdf_pages(pdf_path)
    logger.info("Rendered %d pages", len(pages))

    all_chunks = []
    chunk_index = 0

    for page_num, page_image in enumerate(pages):
        items = _analyze_page(page_image, page_num, model, processor)

        for item in items:
            chunk_type = "figure" if item.get("type", "").lower() == "figure" else "table"
            description = (
                f"{item.get('number', chunk_type.title())} "
                f"(Page {page_num + 1}): "
                f"{item.get('description', '')} "
                f"{item.get('significance', '')}"
            ).strip()

            chunk = {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "chunk_index": chunk_index,
                "chunk_type": chunk_type,
                "text": description,
                "page_num": page_num + 1,
            }
            all_chunks.append(chunk)
            chunk_index += 1

    logger.info(
        "Extracted %d figures/tables from %s",
        len(all_chunks), pdf_path,
    )
    return all_chunks


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.vision <pdf_path> [paper_id]")
        sys.exit(1)

    pdf = sys.argv[1]
    pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    results = extract_figures_tables(pdf, paper_id=pid)
    print(json.dumps(results, indent=2))
