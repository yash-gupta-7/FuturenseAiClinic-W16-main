"""Interactive CLI for the AI Admission Copilot.

Usage:
    python -m app.cli "How long can I defer my enrollment?"
    python -m app.cli            # interactive REPL
"""
import sys

from app import config
from app.pipeline import ask


def _print_result(result):
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])
    print("\nSources retrieved :", ", ".join(dict.fromkeys(result["sources"])))
    print(f"Latency           : {result['latency_s']}s")
    print(f"Tokens            : {result['token_usage']}")
    print(f"Est. cost         : ${result['cost_usd']}")
    print("=" * 70 + "\n")


def main():
    print(f"AI Admission Copilot — config: {config.summary()}")
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _print_result(ask(question))
        return

    print("Ask a question (empty line or Ctrl-C to quit).")
    try:
        while True:
            question = input("\n> ").strip()
            if not question:
                break
            _print_result(ask(question))
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
