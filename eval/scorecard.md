# Baseline Scorecard — (pending run)

This file is **generated** by `python -m eval.run_ragas`. Run the harness with your
`OPENAI_API_KEY` set and it will be overwritten with the real metric table:

```bash
./run.sh
# or
python -m eval.run_ragas
```

Expected shape (values filled by the run):

| Metric | Score | Target (Week 15) | Pass/Fail |
| :--- | :---: | :---: | :---: |
| faithfulness | … | >= 0.90 | … |
| answer_relevancy | … | >= 0.80 | … |
| context_precision | … | >= 0.70 | … |
| context_recall | … | >= 0.80 | … |
| retrieval_hit_rate | … | >= 0.95 | … |
| abstention_accuracy | … | >= 0.90 | … |
| citation_accuracy | … | >= 0.90 | … |
| avg_latency_s | … | <= 1.5s | … |
| avg_cost_usd | … | <= $0.005 | … |
