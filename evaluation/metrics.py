"""
metrics.py — Information retrieval evaluation metrics.

Implements standard IR metrics: Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).
"""

from typing import Optional


def recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Compute Recall@K.

    Recall@K = |{relevant docs in top-K retrieved}| / |{all relevant docs}|

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: List of relevant (ground-truth) document IDs
        k: Cutoff (None = use all retrieved)

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not relevant:
        return 0.0

    top_k = retrieved[:k] if k is not None else retrieved
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)

    return hits / len(relevant_set)


def precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Compute Precision@K.

    Precision@K = |{relevant docs in top-K retrieved}| / K

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: List of relevant (ground-truth) document IDs
        k: Cutoff (None = use all retrieved)

    Returns:
        Precision score between 0.0 and 1.0
    """
    if k is not None:
        top_k = retrieved[:k]
    else:
        top_k = retrieved
        k = len(retrieved)

    if k == 0:
        return 0.0

    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)

    return hits / k


def mrr(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.

    MRR = 1 / rank_of_first_relevant_document

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: List of relevant (ground-truth) document IDs

    Returns:
        Reciprocal rank (0.0 if no relevant doc found)
    """
    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def average_precision(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """
    Compute Average Precision (AP) for a single query.

    AP = (1/|R|) * Σ (Precision@k * rel(k))

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: List of relevant (ground-truth) document IDs

    Returns:
        Average precision score between 0.0 and 1.0
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    hits = 0
    sum_precision = 0.0

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_precision += hits / rank

    return sum_precision / len(relevant_set)


def evaluate_retrieval(
    retrieved: list[str],
    relevant: list[str],
    k_values: list[int] = None,
) -> dict:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: List of relevant (ground-truth) document IDs
        k_values: List of K values for Recall@K and Precision@K

    Returns:
        Dict of metric names to values
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    results = {
        "mrr": mrr(retrieved, relevant),
        "average_precision": average_precision(retrieved, relevant),
    }

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        results[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)

    return results


def evaluate_batch(
    all_retrieved: list[list[str]],
    all_relevant: list[list[str]],
    k_values: list[int] = None,
) -> dict:
    """
    Compute average metrics across multiple queries.

    Args:
        all_retrieved: List of retrieved doc ID lists (one per query)
        all_relevant: List of relevant doc ID lists (one per query)
        k_values: K values for P@K and R@K

    Returns:
        Dict of averaged metric names to values
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    n = len(all_retrieved)
    if n == 0:
        return {}

    aggregated = {}
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        single_results = evaluate_retrieval(retrieved, relevant, k_values)
        for key, value in single_results.items():
            aggregated.setdefault(key, []).append(value)

    # Average across queries
    return {key: sum(values) / len(values) for key, values in aggregated.items()}
