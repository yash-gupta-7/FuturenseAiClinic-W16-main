.PHONY: install ingest eval experiments gate ask clean

install:
	python3 -m pip install -r requirements.txt

ingest:
	python3 -m app.ingest

eval:
	python3 -m eval.run_ragas

experiments:
	python3 -m experiments.run_experiments

gate:
	pytest eval/test_faithfulness_gate.py -v

ask:
	python3 -m app.cli

clean:
	rm -rf .chroma .chroma_* __pycache__ */__pycache__ .deepeval_cache.json
