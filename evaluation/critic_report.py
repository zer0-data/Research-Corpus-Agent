"""
critic_report.py — Generate failure summary reports from critic logs.

Reads critic verdicts from the observability database and produces
a markdown report of common failures, worst queries, and trends.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from src.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)


def generate_critic_report(
    obs_logger: ObservabilityLogger = None,
    output_path: str = "data/critic_report.md",
) -> str:
    """
    Generate a markdown failure summary from critic verdicts.

    Args:
        obs_logger: ObservabilityLogger instance (creates default if None)
        output_path: Path to save the markdown report

    Returns:
        The generated markdown report string
    """
    if obs_logger is None:
        obs_logger = ObservabilityLogger()

    verdicts = obs_logger.get_all_verdicts()

    if not verdicts:
        report = "# Critic Report\n\nNo critic verdicts found in the database."
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
        return report

    # ── Aggregate Statistics ────────────────────────────────────────────

    total = len(verdicts)
    verdict_counts = Counter(v["verdict"] for v in verdicts)
    pass_rate = verdict_counts.get("pass", 0) / total * 100

    # Issue type frequency
    issue_type_counts = Counter()
    issue_severity_counts = Counter()
    all_issues = []

    for v in verdicts:
        for issue in v.get("issues", []):
            if isinstance(issue, dict):
                issue_type_counts[issue.get("type", "unknown")] += 1
                issue_severity_counts[issue.get("severity", "unknown")] += 1
                all_issues.append(issue)
            elif isinstance(issue, str):
                issue_type_counts["untyped"] += 1
                all_issues.append({"description": issue, "type": "untyped"})

    # Average confidence
    confidences = [v["confidence"] for v in verdicts if v["confidence"] is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Worst queries (lowest confidence)
    sorted_verdicts = sorted(verdicts, key=lambda v: v.get("confidence", 1.0))
    worst_queries = sorted_verdicts[:10]

    # ── Build Report ───────────────────────────────────────────────────

    lines = [
        "# Critic Report",
        f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total evaluations | {total} |",
        f"| Pass rate | {pass_rate:.1f}% |",
        f"| Average confidence | {avg_confidence:.3f} |",
        f"| Verdicts — Pass | {verdict_counts.get('pass', 0)} |",
        f"| Verdicts — Partial | {verdict_counts.get('partial', 0)} |",
        f"| Verdicts — Fail | {verdict_counts.get('fail', 0)} |",
        "",
        "## Issue Type Breakdown",
        "",
        "| Issue Type | Count |",
        "|-----------|-------|",
    ]

    for issue_type, count in issue_type_counts.most_common():
        lines.append(f"| {issue_type} | {count} |")

    lines.extend([
        "",
        "## Issue Severity Distribution",
        "",
        "| Severity | Count |",
        "|----------|-------|",
    ])

    for severity, count in issue_severity_counts.most_common():
        lines.append(f"| {severity} | {count} |")

    lines.extend([
        "",
        "## Worst Queries (Lowest Confidence)",
        "",
    ])

    for i, v in enumerate(worst_queries, 1):
        issues_str = "; ".join(
            iss.get("description", str(iss)) if isinstance(iss, dict) else str(iss)
            for iss in v.get("issues", [])
        ) or "No issues recorded"

        lines.extend([
            f"### {i}. Confidence: {v.get('confidence', 'N/A')} — Verdict: {v.get('verdict', 'N/A')}",
            f"- **Query ID**: {v.get('query_id', 'unknown')}",
            f"- **Issues**: {issues_str}",
            f"- **Answer excerpt**: {v.get('answer_excerpt', 'N/A')[:200]}...",
            "",
        ])

    # Most common issues (descriptions)
    lines.extend([
        "## Most Common Issue Descriptions",
        "",
    ])

    desc_counts = Counter()
    for issue in all_issues:
        desc = issue.get("description", str(issue)) if isinstance(issue, dict) else str(issue)
        desc_counts[desc[:100]] += 1

    for desc, count in desc_counts.most_common(10):
        lines.append(f"- **({count}x)** {desc}")

    report = "\n".join(lines)

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")

    logger.info("Critic report saved to %s (%d verdicts analyzed)", output_path, total)
    return report


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    report = generate_critic_report()
    print(report)
