"""KPI targets — the bar every score is judged against.

Provenance: these are anchored to the Week 15 PRD KPIs and the Week 16 Ragas
reference targets. Each entry records where the number came from so it can be
defended in the viva.
"""

# metric_key -> (target, "higher"|"lower", provenance)
TARGETS = {
    # Core Ragas metrics (Week 16 reference targets; faithfulness anchored to the
    # Week 15 PRD "Hallucination Rate = 0%" KPI, operationalised as faithfulness >= 0.90).
    "faithfulness":      (0.90, "higher", "Week15 PRD: Hallucination 0% -> faithfulness >= 0.90 to ship"),
    "answer_relevancy":  (0.80, "higher", "Week16 reference target"),
    "context_precision": (0.70, "higher", "Week16 reference target"),
    "context_recall":    (0.80, "higher", "Week16 reference target"),

    # Operational KPIs carried over from the Week 15 PRD.
    "retrieval_hit_rate": (0.95, "higher", "Week15 PRD: Retrieval Hit Rate @ K=3 >= 95%"),
    "abstention_accuracy": (0.90, "higher", "Week16 Tier B: correct 'I don't know' on out-of-corpus"),
    "citation_accuracy":  (0.90, "higher", "Week15 PRD: citation-backed answers / audit trail"),
    "avg_latency_s":      (1.5, "lower", "Week15 PRD: End-to-end latency < 1.5s"),
    "avg_cost_usd":       (0.005, "lower", "Week15 PRD: Query cost < $0.005"),
}


def passed(metric_key: str, value) -> bool:
    if metric_key not in TARGETS or value is None:
        return False
    target, direction, _ = TARGETS[metric_key]
    return value >= target if direction == "higher" else value <= target
