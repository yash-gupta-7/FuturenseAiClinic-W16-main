"""OPTIONAL: draft a synthetic golden set with Ragas, then HAND-VERIFY.

The committed eval/golden_set.jsonl was hand-authored against the corpus. This
script shows the Ragas synthetic-generation path the spec mentions — useful to
expand the set later. Raw synthetic data is NOT trustworthy: every generated row
must be reviewed and corrected before it enters golden_set.jsonl.

Usage:
    python -m eval.generate_golden --n 10 --out eval/golden_synthetic_draft.jsonl
"""
import os
import json
import argparse

from app import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="eval/golden_synthetic_draft.jsonl")
    args = ap.parse_args()

    if not config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not set — add it to .env.")

    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    docs = DirectoryLoader(
        config.DOCS_DIR, glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()

    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(ChatOpenAI(model=config.EVAL_MODEL, temperature=0,
                                           api_key=config.OPENAI_API_KEY)),
        embedding_model=LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)),
    )
    testset = generator.generate_with_langchain_docs(docs, testset_size=args.n)
    df = testset.to_pandas()

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    with open(out, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({
                "question": r.get("user_input") or r.get("question"),
                "ground_truth": r.get("reference") or r.get("ground_truth"),
                "flavor": "UNVERIFIED",
                "expected_sources": [],
            }) + "\n")
    print(f"Wrote {len(df)} UNVERIFIED draft rows to {out}.")
    print("HAND-VERIFY each row (fix the answer, set flavor, set expected_sources) "
          "before merging into golden_set.jsonl.")


if __name__ == "__main__":
    main()
