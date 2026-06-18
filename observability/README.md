# Observability — Langfuse Tracing

The RAG pipeline is wrapped with Langfuse's `@observe()` decorator (see
[`app/observability.py`](../app/observability.py), [`app/pipeline.py`](../app/pipeline.py),
[`app/retriever.py`](../app/retriever.py)). Every call to `answer_question`
produces a trace with the retrieval span, the generation span, token usage,
latency, and an estimated cost, plus the active config as metadata.

## Setup (≈3 minutes)

1. Create a free account at [langfuse.com](https://langfuse.com) (cloud) or self-host.
2. Create a project and copy the **public** and **secret** API keys.
3. Add them to your `.env` (see [`.env.example`](../.env.example)):
   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com   # or https://us.cloud.langfuse.com
   ```
4. Run anything that calls the pipeline:
   ```bash
   python -m app.cli "How long can I defer my enrollment?"
   # or the full eval, which traces all 40 questions:
   python -m eval.run_ragas
   ```
5. Open the Langfuse dashboard — traces appear in real time.

> If Langfuse keys are absent, `@observe()` becomes a no-op and the pipeline still
> runs. Tracing is additive, never required to produce a scorecard.

## What to capture for submission

Drop two screenshots into [`screenshots/`](screenshots/) and reference them in the
root `README.md`:

| File | What it should show |
| :--- | :--- |
| `screenshots/trace.png` | A single trace expanded: the `rag_pipeline` root with the `retrieve` and generation children, token counts, latency, and the metadata block (sources, cost, config). |
| `screenshots/dashboard.png` | The project dashboard: trace volume, latency distribution, and cost over the eval run. |

## Tier A: latency & cost reading

The per-question `latency_s` and `cost_usd` are also written to
`eval/results_raw.jsonl` and averaged into `eval/scorecard.md`
(`avg_latency_s`, `avg_cost_usd`). Cross-check those numbers against the Langfuse
dashboard's latency/cost panels and note the comparison in the exec memo.
