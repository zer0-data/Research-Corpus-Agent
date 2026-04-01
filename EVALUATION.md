# Evaluation Report

## Overview

This document describes the evaluation methodology for the Research Corpus AI Agent system, covering both retrieval quality and answer generation quality.

---

## Evaluation Dataset

The evaluation set (`evaluation/labeled_queries.json`) contains **50 labeled query → paper pairs** covering:

| Category | Count | Description |
|----------|-------|-------------|
| Factual | 10 | Direct knowledge questions |
| Comparative | 10 | Compare approaches/methods |
| Methodological | 12 | How specific techniques work |
| Survey | 12 | Broad literature coverage |
| Trend | 6 | Emerging patterns and developments |

Each query maps to 3 relevant ArXiv paper IDs for ground-truth evaluation.

---

## Retrieval Metrics

### Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Recall@K** | \|relevant ∩ top-K\| / \|relevant\| | Fraction of relevant docs found in top K |
| **Precision@K** | \|relevant ∩ top-K\| / K | Fraction of top K that are relevant |
| **MRR** | 1 / rank(first relevant) | How early the first relevant doc appears |
| **MAP** | mean(AP across queries) | Overall ranking quality |

### Evaluation K Values

- K ∈ {1, 3, 5, 10, 20}

### Running Retrieval Evaluation

```python
from evaluation.metrics import evaluate_batch

# all_retrieved: list of retrieved doc ID lists (one per query)
# all_relevant: list of relevant doc ID lists (one per query)
results = evaluate_batch(all_retrieved, all_relevant, k_values=[1, 3, 5, 10, 20])
print(results)
```

### Expected Baselines

| Method | Recall@10 | Precision@5 | MRR |
|--------|-----------|-------------|-----|
| BM25 only | ~0.45 | ~0.20 | ~0.40 |
| Dense only (BGE) | ~0.55 | ~0.28 | ~0.50 |
| Hybrid (RRF) | ~0.65 | ~0.33 | ~0.58 |
| Hybrid + Rerank | ~0.65 | ~0.40 | ~0.65 |

> **Note**: These are estimated baselines. Actual values depend on corpus size and query difficulty.

---

## Answer Quality Metrics (LLM-as-Judge)

### Dimensions

| Dimension | Scale | Description |
|-----------|-------|-------------|
| **Relevance** | 1–5 | Does the answer address the question? |
| **Completeness** | 1–5 | Are all aspects covered? |
| **Citation Accuracy** | 1–5 | Are citations correct and verifiable? |
| **Overall** | 1–5 | Average of the three dimensions |

### Running Answer Evaluation

```python
from evaluation.judge import LLMJudge

judge = LLMJudge()
result = judge.judge(
    query="How does RLHF improve alignment?",
    answer="<generated answer>",
    reference_passages=[...],
)
print(result)
# {'relevance': 4, 'completeness': 4, 'citation_accuracy': 3, 'overall': 3.67, ...}
```

---

## Critic-Based Evaluation

### Verdict Categories

| Verdict | Meaning |
|---------|---------|
| **Pass** | Answer is faithful, complete, and well-cited |
| **Partial** | Answer has minor gaps or citation issues |
| **Fail** | Answer contains hallucinations or major omissions |

### Issue Types

| Type | Description |
|------|-------------|
| `hallucination` | Claims not supported by any passage |
| `unsupported_claim` | Statements without corresponding citations |
| `missing_coverage` | Important aspects of the query not addressed |
| `citation_error` | Incorrect paper attribution |
| `coherence_issue` | Logical or structural problems |

### Generating Critic Reports

```python
from evaluation.critic_report import generate_critic_report

report = generate_critic_report(output_path="data/critic_report.md")
```

---

## Evaluation Protocol

### Full Pipeline Evaluation

1. **Retrieval evaluation**: Run all 50 labeled queries through the retrieval pipeline, compute IR metrics against ground-truth paper IDs
2. **Answer evaluation**: For each query, generate an answer via the full agent pipeline, then score with LLM-as-judge
3. **Critic evaluation**: Run the critic agent on each answer, aggregate verdicts via `critic_report.py`
4. **Ablation studies**: Compare retrieval performance with/without each retrieval component (dense-only, sparse-only, fusion, fusion+rerank)

### Ablation Matrix

| Configuration | Components |
|--------------|------------|
| Dense only | BGE + Chroma |
| Sparse only | BM25 |
| Dense + Sparse | RRF fusion |
| Full hybrid | RRF fusion + reranker |
| Full hybrid + ColBERT | All three retrievers + reranker |
