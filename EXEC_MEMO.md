# Executive Memo — AI Admission Copilot: The Evaluable Core

**To:** Course Review Panel / Engineering Leadership
**From:** Het Patel (Applied ML Engineering)
**Date:** June 18, 2026
**Re:** Week 16 — a RAG slice whose quality is a measured number

> One-page, interview-ready. This is my answer to "tell me about a time you
> evaluated an AI system." Bracketed `[fill]` values are populated from the real
> run (`./run.sh`) — they are deliberately not faked.

---

## 1. What I built

A thin vertical slice of the AI Admission Copilot, end to end:
**ingest → chunk → embed (`text-embedding-3-small`) → Chroma → retrieve top-k →
grounded prompt → LLM answer with citations**, over a corpus of **17 admission-policy
documents**. Bolted on is the part that matters: an **evaluation + observability
harness**. A reviewer runs one command (`./run.sh`) and reproduces a scorecard;
every pipeline call is traced in Langfuse with tokens, latency, and cost.

The bar is not arbitrary — it is the KPI set from my Week 15 PRD (retrieval
hit-rate ≥ 95%, latency < 1.5s, cost < $0.005/query, zero hallucination →
faithfulness ≥ 0.90).

## 2. How I measure it

A **40-question golden set**, hand-authored and corpus-verified, spanning four
flavors: easy, ambiguous (vocabulary-overlap traps), multi-hop (two-document), and
adversarial (out-of-corpus → must abstain). Scored with **Ragas** on the four core
metrics, plus retrieval hit-rate, abstention accuracy, citation accuracy, and
latency/cost pulled per query.

## 3. Baseline scorecard `[fill from eval/scorecard.md]`

| Metric | Score | Target | Verdict |
| :--- | :---: | :---: | :---: |
| Faithfulness | `[ ]` | ≥ 0.90 | `[ ]` |
| Answer relevancy | `[ ]` | ≥ 0.80 | `[ ]` |
| Context precision | `[ ]` | ≥ 0.70 | `[ ]` |
| Context recall | `[ ]` | ≥ 0.80 | `[ ]` |
| Retrieval hit-rate @3 | `[ ]` | ≥ 0.95 | `[ ]` |
| Abstention accuracy | `[ ]` | ≥ 0.90 | `[ ]` |
| Avg latency / cost | `[ ]` | <1.5s / <$0.005 | `[ ]` |

## 4. The experiment that moved a metric most `[fill]`

I changed one variable at a time on the same golden set. `[State the winner, e.g.
"top-k 3→5 raised context_recall +0.0X for +Yms latency"]`, and I **`[kept /
reverted]`** it because `[reason tied to the binding constraint]`. The losing
change: `[summary]`. (Full deltas: `experiments/RESULTS.md`.)

## 5. Honest read: what is and isn't production-ready

**Ready:** the slice runs end to end, abstains on out-of-corpus questions, cites
sources, and is fully traced and gated — a faithfulness regression below 0.70
hard-fails the build (DeepEval).

**Not ready:** `[fill from results — e.g. the Week 15 ambiguous failure modes
("fee waiver" vs "financial aid") if they persist]`. Before production I would
add: hybrid (BM25 + dense) retrieval and a reranker (the Week 15 recommendation,
now testable against this harness), a freshness/re-index plan for when policies
change each term, monitoring on faithfulness and latency in Langfuse with alerts,
and guardrails on the abstention path. The point of this week is that I can now
make each of those claims with a number, not a hope.
