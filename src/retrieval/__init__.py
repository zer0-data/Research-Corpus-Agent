"""
retrieval — Hybrid retrieval system for the ArXiv Research Corpus Agent.

Provides the HybridRetriever pipeline:
  DenseRetriever + SparseRetriever + (optional) ColBERTRetriever
  → HybridFuser (RRF)
  → CrossEncoderReranker

All models run locally on T4 GPU.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.retrieval.dense import DenseRetriever, ColBERTRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.fusion import HybridFuser
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

USE_COLBERT: bool = True  # Set False for 2-way RRF fallback (memory pressure)


# ── HybridRetriever Pipeline ────────────────────────────────────────────────

class HybridRetriever:
    """
    End-to-end hybrid retrieval pipeline.

    1. Runs DenseRetriever, SparseRetriever, and (optionally) ColBERTRetriever
       in parallel via ThreadPoolExecutor
    2. Fuses results with Reciprocal Rank Fusion (HybridFuser)
    3. Re-ranks the fused list with CrossEncoderReranker

    Config:
        use_colbert: If True, include ColBERT in 3-way RRF.
                     If False, run 2-way RRF (BGE + BM25 only).
    """

    def __init__(
        self,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        colbert: Optional[ColBERTRetriever] = None,
        fuser: Optional[HybridFuser] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        use_colbert: bool = USE_COLBERT,
        obs_logger=None,
    ):
        """
        Args:
            dense: DenseRetriever instance (created lazily if None)
            sparse: SparseRetriever instance (created lazily if None)
            colbert: ColBERTRetriever instance (created lazily if use_colbert)
            fuser: HybridFuser instance (created with defaults if None)
            reranker: CrossEncoderReranker instance (created lazily if None)
            use_colbert: Whether to include ColBERT retriever
            obs_logger: ObservabilityLogger instance for logging (optional)
        """
        self.use_colbert = use_colbert
        self.obs_logger = obs_logger

        # Initialize components (lazy — only create if not provided)
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()
        self.fuser = fuser or HybridFuser()
        self.reranker = reranker or CrossEncoderReranker()

        self.colbert = None
        if self.use_colbert:
            if colbert is not None:
                self.colbert = colbert
            else:
                try:
                    self.colbert = ColBERTRetriever()
                except Exception as e:
                    logger.warning(
                        "ColBERT init failed (%s). Falling back to 2-way RRF.", e
                    )
                    self.use_colbert = False

        active = ["dense", "sparse"]
        if self.use_colbert:
            active.append("colbert")
        logger.info("HybridRetriever ready — active retrievers: %s", active)

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        """
        Run the full hybrid retrieval pipeline.

        1. DenseRetriever, SparseRetriever, ColBERTRetriever in parallel
        2. HybridFuser (RRF)
        3. CrossEncoderReranker

        Args:
            query: Search query string
            top_n: Number of final results to return

        Returns:
            Re-ranked list of top_n result dicts
        """
        dense_results = []
        sparse_results = []
        colbert_results = []

        active_retrievers = []

        # Run retrievers in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            futures[executor.submit(self.dense.retrieve, query, 50)] = "dense"
            active_retrievers.append("dense")

            futures[executor.submit(self.sparse.retrieve, query, 50)] = "sparse"
            active_retrievers.append("sparse")

            if self.use_colbert and self.colbert is not None:
                futures[executor.submit(self.colbert.retrieve, query, 50)] = "colbert"
                active_retrievers.append("colbert")

            for future in as_completed(futures):
                retriever_name = futures[future]
                try:
                    results = future.result()
                    if retriever_name == "dense":
                        dense_results = results
                    elif retriever_name == "sparse":
                        sparse_results = results
                    elif retriever_name == "colbert":
                        colbert_results = results

                    logger.debug(
                        "%s returned %d results", retriever_name, len(results)
                    )
                except Exception as e:
                    logger.error("%s retrieval failed: %s", retriever_name, e)

        # Fuse results
        fused = self.fuser.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            colbert_results=colbert_results if self.use_colbert else None,
        )

        logger.debug("Fusion produced %d unique documents", len(fused))

        # Rerank
        final = self.reranker.rerank(query=query, docs=fused, top_n=top_n)

        # Log which retrievers were active
        if self.obs_logger:
            try:
                self.obs_logger.log_decision(
                    query_id="",
                    agent_name="hybrid_retriever",
                    action="search",
                    input_summary=f"query: {query[:200]}",
                    output_summary=(
                        f"active: {active_retrievers} | "
                        f"dense: {len(dense_results)}, "
                        f"sparse: {len(sparse_results)}, "
                        f"colbert: {len(colbert_results)} | "
                        f"fused: {len(fused)} → reranked top {len(final)}"
                    ),
                    reasoning=f"use_colbert={self.use_colbert}",
                )
            except Exception:
                pass  # Don't fail search if logging fails

        logger.info(
            "HybridRetriever: %s → %d fused → %d reranked (query: %s)",
            active_retrievers, len(fused), len(final), query[:80],
        )

        return final
