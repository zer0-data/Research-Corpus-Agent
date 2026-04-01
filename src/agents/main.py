"""
main.py — Multi-agent orchestrator for the ArXiv Research Corpus Agent.

Wires PlannerAgent → RetrieverAgent → AnalystAgent → CriticAgent
with a critic-driven revision loop (max 2 retries).

Logs every step via ObservabilityLogger.
"""

import json
import logging
import sys

from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)


def run_agent_pipeline(
    query: str,
    retriever_agent: RetrieverAgent = None,
    obs_logger: ObservabilityLogger = None,
    max_revisions: int = 2,
) -> dict:
    """
    Run the full multi-agent pipeline.

    Flow:
    1. PlannerAgent.plan(query) → sub_queries
    2. RetrieverAgent.retrieve(sub_queries) → docs
    3. AnalystAgent.analyze(query, docs) → answer
    4. CriticAgent.critique(query, answer, docs) → critique
    5. If critique.verdict == "revise" and retries < max_revisions:
           AnalystAgent re-analyzes with critic feedback, then re-critique
    6. Return final structured output

    Args:
        query: User research question
        retriever_agent: Pre-configured RetrieverAgent (created if None)
        obs_logger: ObservabilityLogger for tracing (created if None)
        max_revisions: Maximum number of analyst revision loops

    Returns:
        Dict with: answer, sources, sub_queries, critic_verdict, failure_type
    """
    obs = obs_logger or ObservabilityLogger()

    planner = PlannerAgent()
    retriever = retriever_agent or RetrieverAgent()
    analyst = AnalystAgent()
    critic = CriticAgent()

    # Log the query
    query_id = obs.log_query(query)

    # ── Step 1: Plan ─────────────────────────────────────────────────────
    logger.info("Step 1: Planning sub-queries...")
    sub_queries = planner.plan(query)

    obs.log_decision(
        query_id=query_id,
        agent_name="planner",
        action="plan",
        input_summary=query[:500],
        output_summary=f"{len(sub_queries)} sub-queries",
        reasoning=json.dumps(sub_queries),
    )

    # ── Step 2: Retrieve ─────────────────────────────────────────────────
    logger.info("Step 2: Retrieving documents...")
    docs = retriever.retrieve(sub_queries)

    obs.log_decision(
        query_id=query_id,
        agent_name="retriever",
        action="retrieve",
        input_summary=f"{len(sub_queries)} sub-queries",
        output_summary=f"{len(docs)} unique documents",
        reasoning="Hybrid search: dense + sparse + ColBERT → RRF → reranker",
    )

    obs.log_retrieval(
        query_id=query_id,
        sub_query="; ".join(sub_queries),
        retriever_type="hybrid_reranked",
        doc_ids=[d.get("doc_id", "") for d in docs],
        scores=[d.get("rerank_score", d.get("rrf_score", 0.0)) for d in docs],
    )

    # ── Step 3: Analyze ──────────────────────────────────────────────────
    logger.info("Step 3: Synthesizing answer...")
    answer = analyst.analyze(query, docs)

    obs.log_decision(
        query_id=query_id,
        agent_name="analyst",
        action="synthesize",
        input_summary=f"Query + {len(docs)} docs",
        output_summary=f"Answer: {len(answer)} chars",
        reasoning="Initial synthesis",
    )

    # ── Step 4: Critique + Revision Loop ─────────────────────────────────
    critique = None
    for revision in range(max_revisions + 1):
        logger.info("Step 4: Critic evaluation (round %d)...", revision + 1)
        critique = critic.critique(query, answer, docs)

        obs.log_decision(
            query_id=query_id,
            agent_name="critic",
            action="critique",
            input_summary=f"Answer ({len(answer)} chars) + {len(docs)} docs",
            output_summary=f"verdict={critique['verdict']}, failure={critique['failure_type']}",
            reasoning=json.dumps(critique.get("issues", [])),
        )

        obs.log_verdict(
            query_id=query_id,
            verdict=critique["verdict"],
            confidence=1.0 if critique["verdict"] == "pass" else 0.5,
            issues=critique.get("issues", []),
            answer_excerpt=answer[:500],
        )

        if critique["verdict"] == "pass":
            logger.info("Critic passed on round %d", revision + 1)
            break

        if revision < max_revisions:
            # Revise: re-analyze with critic feedback appended
            logger.info("Critic requested revision — re-analyzing...")
            feedback = "; ".join(critique.get("issues", []))
            augmented_query = f"{query}\nCritic feedback: {feedback}"
            answer = analyst.analyze(augmented_query, docs)

            obs.log_decision(
                query_id=query_id,
                agent_name="analyst",
                action="revise",
                input_summary=f"Augmented query + {len(docs)} docs (revision {revision + 1})",
                output_summary=f"Revised answer: {len(answer)} chars",
                reasoning=f"Feedback: {feedback[:200]}",
            )
        else:
            # Use revised_answer from critic if provided
            if critique.get("revised_answer"):
                answer = critique["revised_answer"]
                logger.info("Using critic's revised answer")

    # ── Build Final Output ───────────────────────────────────────────────
    sources = [
        {
            "doc_id": d.get("doc_id", ""),
            "title": d.get("metadata", {}).get("title", d.get("title", "")),
            "paper_id": d.get("metadata", {}).get("paper_id", ""),
            "score": d.get("rerank_score", d.get("rrf_score", 0.0)),
        }
        for d in docs[:10]
    ]

    output = {
        "answer": answer,
        "sources": sources,
        "sub_queries": sub_queries,
        "critic_verdict": critique["verdict"] if critique else "pass",
        "failure_type": critique.get("failure_type") if critique else None,
    }

    logger.info(
        "Pipeline complete — verdict: %s, failure: %s, sources: %d",
        output["critic_verdict"],
        output["failure_type"],
        len(output["sources"]),
    )

    return output


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.main \"your research question\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = run_agent_pipeline(query)

    print("\n" + "=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(result["answer"])
    print("\n" + "-" * 70)
    print(f"Sub-queries: {result['sub_queries']}")
    print(f"Sources: {len(result['sources'])}")
    print(f"Critic verdict: {result['critic_verdict']}")
    print(f"Failure type: {result['failure_type']}")
    print("-" * 70)
    print(json.dumps(result, indent=2, default=str))
