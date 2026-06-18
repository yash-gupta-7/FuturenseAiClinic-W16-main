"""Run the RAG pipeline over the golden set and produce a Ragas scorecard.

Usage:
    python -m eval.run_ragas                 # full run, writes eval/scorecard.md
    python -m eval.run_ragas --limit 5       # quick smoke test on 5 questions
    python -m eval.run_ragas --tag baseline  # label the run (used by experiments)

Outputs:
    eval/results_raw.jsonl   per-question answers, contexts, latency, cost
    eval/scorecard.md        metric -> score -> Week15 target -> pass/fail
"""
import os
import json
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Any

from app import config
from app.ingest import build_index, load_index
from app.pipeline import answer_question
from app.observability import flush
from eval.targets import TARGETS, passed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(ROOT, "eval", "golden_set.jsonl")
RAW_PATH = os.path.join(ROOT, "eval", "results_raw.jsonl")
SCORECARD_PATH = os.path.join(ROOT, "eval", "scorecard.md")


def load_golden(limit: int = None) -> List[Dict[str, Any]]:
    rows = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def run_pipeline_over_golden(rows, store, top_k) -> List[Dict[str, Any]]:
    """Execute the RAG pipeline for each golden question; collect everything."""
    records = []
    for i, row in enumerate(rows, 1):
        res = answer_question(row["question"], store=store, top_k=top_k)
        records.append({**row, **res})
        print(f"  [{i}/{len(rows)}] {row['flavor']:11s} {row['question'][:60]}")
    flush()
    return records


# --- custom (non-Ragas) metrics -------------------------------------------

def retrieval_hit_rate(records) -> float:
    """Fraction of answerable questions where an expected source was retrieved."""
    answerable = [r for r in records if r["flavor"] != "adversarial"]
    if not answerable:
        return None
    hits = 0
    for r in answerable:
        retrieved = set(r["sources"])
        if any(src in retrieved for src in r["expected_sources"]):
            hits += 1
    return hits / len(answerable)


def abstention_accuracy(records) -> float:
    """Fraction of adversarial questions the system correctly abstained on."""
    adversarial = [r for r in records if r["flavor"] == "adversarial"]
    if not adversarial:
        return None
    correct = sum(1 for r in adversarial if _is_abstention(r["answer"]))
    return correct / len(adversarial)


def _is_abstention(answer: str) -> bool:
    a = answer.lower()
    return "i don't know" in a or "i do not know" in a


def citation_accuracy(records) -> float:
    """For answerable, non-abstained answers: did the cited file(s) appear in the
    retrieved context (i.e. the citation is grounded, not invented)?"""
    import re
    scored = []
    for r in records:
        if r["flavor"] == "adversarial" or _is_abstention(r["answer"]):
            continue
        cited = set(re.findall(r"\[([a-zA-Z0-9_\-]+\.(?:md|txt|pdf))\]", r["answer"]))
        if not cited:
            scored.append(0.0)  # made a claim but cited nothing
            continue
        retrieved = set(r["sources"])
        grounded = sum(1 for c in cited if c in retrieved)
        scored.append(grounded / len(cited))
    return statistics.mean(scored) if scored else None


# --- Ragas core metrics ----------------------------------------------------

def compute_ragas(records) -> Dict[str, float]:
    """Compute the four core Ragas metrics on the answerable subset."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    answerable = [r for r in records if r["flavor"] != "adversarial"]

    samples = [{
        "user_input": r["question"],
        "retrieved_contexts": r["contexts"],
        "response": r["answer"],
        "reference": r["ground_truth"],
    } for r in answerable]

    dataset = EvaluationDataset.from_list(samples)

    judge = LangchainLLMWrapper(ChatOpenAI(model=config.EVAL_MODEL, temperature=0,
                                           api_key=config.OPENAI_API_KEY))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    )

    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()

    # Map Ragas' column names to our canonical keys.
    colmap = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "response_relevancy": "answer_relevancy",
        "llm_context_precision_with_reference": "context_precision",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
        "llm_context_recall": "context_recall",
    }
    scores = {}
    for col in df.columns:
        key = colmap.get(col)
        if key:
            series = df[col].dropna()
            if len(series):
                scores[key] = float(series.mean())
    return scores


# --- scorecard writer ------------------------------------------------------

ORDER = [
    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    "retrieval_hit_rate", "abstention_accuracy", "citation_accuracy",
    "avg_latency_s", "avg_cost_usd",
]


def write_scorecard(scores, n_total, n_answerable, n_adversarial, tag):
    lines = []
    lines.append(f"# Baseline Scorecard — {tag}")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_  ")
    lines.append(f"_Config: {json.dumps(config.summary())}_  ")
    lines.append(f"_Golden set: {n_total} questions ({n_answerable} answerable, "
                 f"{n_adversarial} adversarial)_")
    lines.append("")
    lines.append("| Metric | Score | Target (Week 15) | Pass/Fail |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for key in ORDER:
        if key not in scores or scores[key] is None:
            continue
        target, direction, _prov = TARGETS[key]
        val = scores[key]
        ok = passed(key, val)
        comp = ">=" if direction == "higher" else "<="
        if key in ("avg_latency_s",):
            sval, starget = f"{val:.3f}s", f"{comp} {target}s"
        elif key in ("avg_cost_usd",):
            sval, starget = f"${val:.5f}", f"{comp} ${target}"
        else:
            sval, starget = f"{val:.3f}", f"{comp} {target:.2f}"
        lines.append(f"| {key} | {sval} | {starget} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append("### Target provenance")
    for key in ORDER:
        if key in scores and scores[key] is not None:
            lines.append(f"- **{key}**: {TARGETS[key][2]}")
    lines.append("")
    with open(SCORECARD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nScorecard written to {SCORECARD_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--rebuild", action="store_true",
                    help="Rebuild the Chroma index before evaluating.")
    ap.add_argument("--no-ragas", action="store_true",
                    help="Skip the LLM-judged Ragas metrics (custom metrics only).")
    args = ap.parse_args()

    if not config.OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    store = build_index() if args.rebuild else load_index()
    rows = load_golden(args.limit)

    print(f"Running pipeline over {len(rows)} golden questions (top_k={args.top_k})...")
    records = run_pipeline_over_golden(rows, store, args.top_k)

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Raw results written to {RAW_PATH}")

    scores = {}
    if not args.no_ragas:
        print("\nComputing Ragas metrics (LLM-judged)...")
        scores.update(compute_ragas(records))

    scores["retrieval_hit_rate"] = retrieval_hit_rate(records)
    scores["abstention_accuracy"] = abstention_accuracy(records)
    scores["citation_accuracy"] = citation_accuracy(records)
    lat = [r["latency_s"] for r in records]
    cost = [r["cost_usd"] for r in records]
    scores["avg_latency_s"] = statistics.mean(lat) if lat else None
    scores["avg_cost_usd"] = statistics.mean(cost) if cost else None

    n_adv = sum(1 for r in records if r["flavor"] == "adversarial")
    write_scorecard(scores, len(records), len(records) - n_adv, n_adv, args.tag)


if __name__ == "__main__":
    main()
