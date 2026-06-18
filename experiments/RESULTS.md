# Experiment Results — (pending run)

This file is **generated** by `python -m experiments.run_experiments`. It runs the
baseline and each one-variable experiment over the same golden set and tabulates
the metric deltas. Run:

```bash
python -m experiments.run_experiments
```

Experiments included:

1. **top-k**: 3 (baseline) → 5
2. **chunk size**: 1000/200 (baseline) → 500/100 (rebuilds the index)

After the run, fill in the **Decision** line: which change you keep and why.
