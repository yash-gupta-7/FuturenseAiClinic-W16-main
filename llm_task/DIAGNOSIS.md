# LLM-Integrated Task — Worst-Case Diagnosis (pass/fail gate)

> **Gate rule (from the brief):** take your single worst-scoring question, ask an
> LLM to diagnose why it failed and propose a fix, then **actually try the fix and
> re-measure**. A diagnosis with no attempted fix and re-measurement fails the gate.

This file is a template. Fill every section **after** you have run
`python -m eval.run_ragas` and identified your lowest-faithfulness (or retrieval-miss)
case from `eval/results_raw.jsonl` and `eval/scorecard.md`.

---

## 1. The failing case

- **Question ID:** `qXXX`
- **Question:** _paste_
- **Expected source(s):** _paste from golden_set.jsonl_
- **Retrieved sources:** _paste from results_raw.jsonl_
- **Retrieved context (abbreviated):**
  ```
  _paste the chunks the retriever returned_
  ```
- **Bad answer produced:**
  ```
  _paste_
  ```
- **Scores:** faithfulness = _x_, answer_relevancy = _x_, context_precision = _x_, context_recall = _x_

---

## 2. The prompt I gave the LLM

```
Here is a failing case from my RAG evaluation.
Question: <...>
Retrieved context: <...>
Model answer: <...>
Ragas scores: <...>

Diagnose the single most likely root cause of this failure. State whether it is a
RETRIEVAL failure (wrong/missing context) or a GENERATION failure (context was
fine but the answer ignored or contradicted it). Then propose one concrete,
testable fix I can make to my pipeline, and predict which metric it should move.
```

## 3. The model's diagnosis (verbatim)

```
_paste the LLM's response_
```

## 4. My critique of the diagnosis

- Was the retrieval-vs-generation call correct? _yes/no — why_
- Did the diagnosis match what the scores actually show (e.g. low context_recall
  points to retrieval; high context_recall + low faithfulness points to generation)? _..._
- Anything the model missed or got wrong? _..._

## 5. The fix I tried and the re-measured result

- **Change made:** _e.g. raised top_k 3 -> 5 / shrank chunk_size / tightened the
  abstention instruction in the system prompt_
- **Command:** `python -m eval.run_ragas --tag fix_qXXX` (or an experiments run)
- **Metric before -> after:** _e.g. faithfulness 0.62 -> 0.94_ on this case;
  _golden-set average X -> Y_
- **Verdict:** _did the fix move the metric? Keep it or revert? Why?_
