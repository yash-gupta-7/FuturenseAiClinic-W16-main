"""Run improvement experiments and report metric deltas vs. baseline.

Each experiment changes exactly ONE variable and re-measures on the SAME golden
set, so any delta is attributable to that change. Results are written to
experiments/RESULTS.md.

Usage:
    python -m experiments.run_experiments               # all experiments
    python -m experiments.run_experiments --limit 12    # faster subset
    python -m experiments.run_experiments --only top_k  # one experiment

Experiments:
    top_k      retrieval breadth: k=3 (baseline) vs k=5
    chunk      chunk_size 1000/overlap200 (baseline) vs 500/100 (rebuilds index)
"""
import os
import json
import argparse
import statistics
from datetime import datetime

from app import config
from app.ingest import build_index, load_index
from eval.run_ragas import (
    load_golden, run_pipeline_over_golden, compute_ragas,
    retrieval_hit_rate, abstention_accuracy, citation_accuracy,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT, "experiments", "RESULTS.md")

CORE = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
EXTRA = ["retrieval_hit_rate", "abstention_accuracy", "citation_accuracy",
         "avg_latency_s", "avg_cost_usd"]


def evaluate(rows, store, top_k, with_ragas=True):
    records = run_pipeline_over_golden(rows, store, top_k)
    scores = {}
    if with_ragas:
        scores.update(compute_ragas(records))
    scores["retrieval_hit_rate"] = retrieval_hit_rate(records)
    scores["abstention_accuracy"] = abstention_accuracy(records)
    scores["citation_accuracy"] = citation_accuracy(records)
    scores["avg_latency_s"] = statistics.mean([r["latency_s"] for r in records])
    scores["avg_cost_usd"] = statistics.mean([r["cost_usd"] for r in records])
    return scores


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def delta_row(name, base, exp):
    cells = [name]
    for k in CORE + EXTRA:
        b, e = base.get(k), exp.get(k)
        if b is None or e is None:
            cells.append("-")
        else:
            cells.append(f"{e:.3f} ({e-b:+.3f})")
    return "| " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", choices=["top_k", "chunk"], default=None)
    ap.add_argument("--no-ragas", action="store_true")
    args = ap.parse_args()

    if not config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not set — add it to .env.")

    with_ragas = not args.no_ragas
    rows = load_golden(args.limit)

    # --- Baseline (k=3, chunk 1000/200) ---
    print("\n=== BASELINE (top_k=3, chunk_size=1000, overlap=200) ===")
    base_store = build_index()  # ensure a clean baseline index
    baseline = evaluate(rows, base_store, top_k=3, with_ragas=with_ragas)

    results = []  # (name, scores)

    # --- Experiment 1: top_k = 5 ---
    if args.only in (None, "top_k"):
        print("\n=== EXPERIMENT 1: top_k = 5 (same index) ===")
        exp1 = evaluate(rows, base_store, top_k=5, with_ragas=with_ragas)
        results.append(("top_k=5", exp1))

    # --- Experiment 2: chunk_size = 500 / overlap = 100 (rebuild) ---
    if args.only in (None, "chunk"):
        print("\n=== EXPERIMENT 2: chunk_size=500, overlap=100 (rebuild index) ===")
        small_dir = os.path.join(ROOT, ".chroma_chunk500")
        small_store = build_index(chunk_size=500, chunk_overlap=100,
                                  persist_directory=small_dir,
                                  collection_name="admission_policies_chunk500")
        exp2 = evaluate(rows, small_store, top_k=3, with_ragas=with_ragas)
        results.append(("chunk=500/100", exp2))

    # --- Write RESULTS.md ---
    header = ["Config"] + CORE + EXTRA
    lines = [
        "# Experiment Results",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_  ",
        f"_Golden questions evaluated: {len(rows)}_  ",
        "",
        "Each experiment changes one variable and re-measures on the same golden "
        "set. Cells show the new score with the delta vs. baseline in parentheses.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---"] + [":---:"] * (len(header) - 1)) + " |",
        "| baseline (k=3, 1000/200) | " + " | ".join(fmt(baseline[k]) for k in CORE + EXTRA) + " |",
    ]
    for name, scores in results:
        lines.append(delta_row(name, baseline, scores))
    lines += [
        "",
        "## Decision",
        "",
        "_TODO after the run: state which change you keep and why. Example: "
        "\"top_k=5 lifted context_recall by +0.0X at a +Yms latency cost; kept "
        "because recall was the binding constraint from the Week 15 spike.\"_",
        "",
    ]
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
