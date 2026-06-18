#!/usr/bin/env bash
# One command to reproduce the baseline scorecard.
# Usage: ./run.sh            (full 40-question run)
#        ./run.sh --limit 8  (quick smoke test)
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: no .env found. Copy .env.example to .env and add OPENAI_API_KEY." >&2
  exit 1
fi

echo "==> Installing dependencies (first run only)..."
python3 -m pip install -q -r requirements.txt

echo "==> Building Chroma index from docs/ ..."
python3 -m app.ingest

echo "==> Running golden-set evaluation + Ragas scorecard ..."
python3 -m eval.run_ragas "$@"

echo ""
echo "Done. See eval/scorecard.md and eval/results_raw.jsonl."
echo "Next: python -m experiments.run_experiments  (metric deltas)"
echo "      pytest eval/test_faithfulness_gate.py -v   (Tier A gate)"
