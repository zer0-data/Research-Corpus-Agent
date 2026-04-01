# Research Corpus AI Agent

> An intelligent multi-agent system for answering complex research queries over a corpus of 100K+ ArXiv CS/AI/ML papers, powered by hybrid retrieval and LLM-based reasoning.

---

## Features

- **Data Pipeline** — Automated ingestion, chunking, and indexing of ArXiv metadata
- **Hybrid Retrieval** — Dense (BGE), Sparse (BM25), and ColBERT retrieval fused via Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — ms-marco-MiniLM-L-12-v2 for precision reranking
- **Multi-Agent Architecture** — Planner → Retriever → Analyst → Critic pipeline
- **Vision Extraction** — Qwen2.5-VL for figure/table extraction from PDFs
- **Full Observability** — SQLite logging of all queries, retrievals, and agent decisions
- **Evaluation Suite** — IR metrics, LLM-as-judge scoring, and critic failure reports

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────┐    ┌────────────┐    ┌───────────┐    ┌──────────┐
│ Planner  │───▶│ Retriever  │───▶│  Analyst  │───▶│  Critic  │
│ (decomp) │    │ (hybrid)   │    │ (synth)   │    │ (verify) │
└──────────┘    └────────────┘    └───────────┘    └──────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │  Dense  │  │  BM25   │  │ ColBERT │
    │ (Chroma)│  │(rank_bm)│  │(pylate) │
    └─────────┘  └─────────┘  └─────────┘
                      │
                      ▼
              ┌──────────────┐
              │   Reranker   │
              │(cross-encoder)│
              └──────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Credentials

```bash
# Kaggle API (for ArXiv dataset download)
# Place your kaggle.json at ~/.kaggle/kaggle.json
# Or set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# HuggingFace API (for LLM inference)
export HF_TOKEN=your_hf_token
```

### 3. Run the Data Pipeline

```bash
# Step 1: Download and filter ArXiv metadata
python -m src.pipeline.ingest

# Step 2: Chunk papers into retrieval units
python -m src.pipeline.chunk

# Step 3: Embed chunks and build indexes
python -m src.pipeline.embed
```

### 4. Query the System

```python
from src.utils.llm_client import LLMClient
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.reranker import Reranker
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.observability.logger import ObservabilityLogger

# Initialize components
llm = LLMClient()
obs = ObservabilityLogger()
dense = DenseRetriever()
sparse = SparseRetriever()
reranker = Reranker()

# Initialize agents
planner = PlannerAgent(llm_client=llm, obs_logger=obs)
retriever = RetrieverAgent(
    dense_retriever=dense,
    sparse_retriever=sparse,
    reranker=reranker,
    obs_logger=obs,
)
analyst = AnalystAgent(llm_client=llm, obs_logger=obs)
critic = CriticAgent(llm_client=llm, obs_logger=obs)

# Run the pipeline
query = "Compare transformer architectures for long-context language modeling"
query_id, sub_tasks = planner.plan(query)
passages = retriever.retrieve(sub_tasks, query_id=query_id)
answer = analyst.analyze(query, passages, query_id=query_id)
verdict = critic.critique(query, answer, passages, query_id=query_id)

print(answer)
print(f"\nVerdict: {verdict['verdict']} (confidence: {verdict['confidence']:.2f})")
```

---

## Project Structure

```
├── src/
│   ├── pipeline/          # Data ingestion and indexing
│   │   ├── ingest.py      # ArXiv metadata download + filter
│   │   ├── chunk.py       # Text chunking
│   │   ├── embed.py       # Embedding + vector store
│   │   └── vision.py      # Figure/table extraction (Qwen2.5-VL)
│   ├── retrieval/         # Search and ranking
│   │   ├── dense.py       # BGE + ChromaDB vector search
│   │   ├── sparse.py      # BM25 keyword search
│   │   ├── fusion.py      # Reciprocal Rank Fusion
│   │   └── reranker.py    # Cross-encoder reranking
│   ├── agents/            # Multi-agent reasoning
│   │   ├── planner.py     # Query decomposition
│   │   ├── retriever.py   # Hybrid search orchestration
│   │   ├── analyst.py     # Answer synthesis
│   │   └── critic.py      # Quality verification
│   ├── observability/     # Logging and tracing
│   │   └── logger.py      # SQLite observability
│   └── utils/             # Shared utilities
│       ├── llm_client.py  # HuggingFace API wrapper
│       └── parsers.py     # JSON parsing with retry
├── evaluation/            # Evaluation framework
│   ├── metrics.py         # IR metrics (Recall@K, P@K, MRR)
│   ├── judge.py           # LLM-as-judge scoring
│   ├── critic_report.py   # Failure analysis reports
│   └── labeled_queries.json  # 50 labeled test queries
├── data/                  # Generated data (gitignored)
├── ARCHITECTURE.md        # Detailed architecture docs
├── EVALUATION.md          # Evaluation methodology
└── requirements.txt       # Python dependencies
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Embedding (BGE) | CPU (slow) | T4 GPU |
| Reranker | CPU | T4 GPU |
| ColBERT Index | CPU (slow) | T4 GPU |
| Vision (Qwen2.5-VL) | — | A100 GPU |
| Disk (indexed corpus) | 10 GB | 50 GB |

---

## License

This project is for research and educational purposes.