"""Tier A: DeepEval pytest gate.

Fails the build when faithfulness drops below the safety FLOOR (0.70 — the
Week 16 "unsafe" line). This is distinct from the 0.90 *ship* target: 0.90 is
the bar to release; 0.70 is the bar below which the build must hard-fail.

It evaluates the answers captured in eval/results_raw.jsonl, so run the eval
harness first:

    python -m eval.run_ragas
    pytest eval/test_faithfulness_gate.py -v

Requires OPENAI_API_KEY (DeepEval's judge defaults to an OpenAI model).
"""
import os
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "eval", "results_raw.jsonl")

FAITHFULNESS_FLOOR = float(os.getenv("FAITHFULNESS_FLOOR", "0.70"))


def _load_answerable():
    if not os.path.exists(RAW_PATH):
        return []
    rows = []
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Faithfulness needs retrieval context; skip abstentions w/ no context.
            if r.get("flavor") != "adversarial" and r.get("contexts"):
                rows.append(r)
    return rows


CASES = _load_answerable()


@pytest.mark.skipif(not CASES, reason="No results_raw.jsonl — run `python -m eval.run_ragas` first.")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set.")
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_faithfulness_above_floor(case):
    from deepeval import assert_test
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    metric = FaithfulnessMetric(threshold=FAITHFULNESS_FLOOR, model="gpt-4o-mini")
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=case["answer"],
        retrieval_context=case["contexts"],
    )
    assert_test(test_case, [metric])
