"""
logger.py — Observability logging to SQLite.

Logs queries, document retrievals, agent decisions, and critic verdicts
to a SQLite database for traceability and evaluation.
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DB_PATH = Path("data/observability.db")
Base = declarative_base()


# ── ORM Models ───────────────────────────────────────────────────────────────

class QueryLog(Base):
    """Log of user queries submitted to the system."""
    __tablename__ = "queries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    sub_tasks_json = Column(Text)  # JSON serialized list of sub-tasks
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RetrievalLog(Base):
    """Log of document retrievals for each query."""
    __tablename__ = "retrievals"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, nullable=False)
    sub_query = Column(Text)
    retriever_type = Column(String)  # dense, sparse, fusion, reranked
    doc_ids_json = Column(Text)     # JSON list of retrieved doc IDs
    scores_json = Column(Text)      # JSON list of scores
    num_results = Column(Integer)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AgentDecisionLog(Base):
    """Log of agent decisions and reasoning traces."""
    __tablename__ = "agent_decisions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, nullable=False)
    agent_name = Column(String)    # planner, retriever, analyst, critic
    action = Column(String)        # plan, retrieve, synthesize, critique
    input_summary = Column(Text)
    output_summary = Column(Text)
    reasoning = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CriticVerdictLog(Base):
    """Log of critic agent verdicts on answer quality."""
    __tablename__ = "critic_verdicts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, nullable=False)
    verdict = Column(String)        # pass, fail, partial
    confidence = Column(Float)
    issues_json = Column(Text)      # JSON list of issue descriptions
    answer_excerpt = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Database Manager ─────────────────────────────────────────────────────────

class ObservabilityLogger:
    """SQLite-based observability logger for the Research Corpus Agent."""

    def __init__(self, db_path: str = str(DB_PATH)):
        """
        Initialize the logger and create tables if needed.

        Args:
            db_path: Path to the SQLite database file
        """
        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        logger.info("ObservabilityLogger initialized at %s", db_path)

    def log_query(self, query_text: str, sub_tasks: list = None) -> str:
        """
        Log a user query.

        Returns:
            query_id for linking subsequent logs
        """
        import json

        session = self.Session()
        try:
            entry = QueryLog(
                query_text=query_text,
                sub_tasks_json=json.dumps(sub_tasks) if sub_tasks else None,
            )
            session.add(entry)
            session.commit()
            query_id = entry.id
            logger.debug("Logged query: %s (id=%s)", query_text[:80], query_id)
            return query_id
        finally:
            session.close()

    def log_retrieval(
        self,
        query_id: str,
        sub_query: str,
        retriever_type: str,
        doc_ids: list[str],
        scores: list[float] = None,
    ):
        """Log a retrieval operation."""
        import json

        session = self.Session()
        try:
            entry = RetrievalLog(
                query_id=query_id,
                sub_query=sub_query,
                retriever_type=retriever_type,
                doc_ids_json=json.dumps(doc_ids),
                scores_json=json.dumps(scores) if scores else None,
                num_results=len(doc_ids),
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()

    def log_decision(
        self,
        query_id: str,
        agent_name: str,
        action: str,
        input_summary: str = "",
        output_summary: str = "",
        reasoning: str = "",
    ):
        """Log an agent decision."""
        session = self.Session()
        try:
            entry = AgentDecisionLog(
                query_id=query_id,
                agent_name=agent_name,
                action=action,
                input_summary=input_summary[:2000],
                output_summary=output_summary[:2000],
                reasoning=reasoning[:2000],
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()

    def log_verdict(
        self,
        query_id: str,
        verdict: str,
        confidence: float,
        issues: list[str] = None,
        answer_excerpt: str = "",
    ):
        """Log a critic verdict."""
        import json

        session = self.Session()
        try:
            entry = CriticVerdictLog(
                query_id=query_id,
                verdict=verdict,
                confidence=confidence,
                issues_json=json.dumps(issues) if issues else None,
                answer_excerpt=answer_excerpt[:2000],
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()

    def get_all_verdicts(self) -> list[dict]:
        """Retrieve all critic verdicts for reporting."""
        import json

        session = self.Session()
        try:
            verdicts = session.query(CriticVerdictLog).all()
            return [
                {
                    "id": v.id,
                    "query_id": v.query_id,
                    "verdict": v.verdict,
                    "confidence": v.confidence,
                    "issues": json.loads(v.issues_json) if v.issues_json else [],
                    "answer_excerpt": v.answer_excerpt,
                    "timestamp": v.timestamp.isoformat() if v.timestamp else None,
                }
                for v in verdicts
            ]
        finally:
            session.close()

    def get_query_trace(self, query_id: str) -> dict:
        """Get the full trace for a specific query."""
        import json

        session = self.Session()
        try:
            query = session.query(QueryLog).filter_by(id=query_id).first()
            retrievals = session.query(RetrievalLog).filter_by(query_id=query_id).all()
            decisions = session.query(AgentDecisionLog).filter_by(query_id=query_id).all()
            verdicts = session.query(CriticVerdictLog).filter_by(query_id=query_id).all()

            return {
                "query": {
                    "text": query.query_text if query else "",
                    "timestamp": query.timestamp.isoformat() if query and query.timestamp else None,
                },
                "retrievals": [
                    {
                        "sub_query": r.sub_query,
                        "retriever_type": r.retriever_type,
                        "num_results": r.num_results,
                    }
                    for r in retrievals
                ],
                "decisions": [
                    {
                        "agent": d.agent_name,
                        "action": d.action,
                        "reasoning": d.reasoning,
                    }
                    for d in decisions
                ],
                "verdicts": [
                    {
                        "verdict": v.verdict,
                        "confidence": v.confidence,
                        "issues": json.loads(v.issues_json) if v.issues_json else [],
                    }
                    for v in verdicts
                ],
            }
        finally:
            session.close()
