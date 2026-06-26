"""
Ask amicus a question from the command line.
"""
from __future__ import annotations
import sys

sys.stdout.reconfigure(encoding="utf-8")

from .generation.generate import answer


def main() -> None:
    question = " ".join(sys.argv[1:])
    if not question:
        print('Usage: python -m src.ask "your question"')
        return

    trace = answer(question)
    print(trace.answer)
    print("\n--- sources ---")
    for c in trace.sources:
        m = c.metadata
        print(f"  {m['case_name']}, {m['citation']} ({m['opinion_type']})")
    print(f"\n--- {trace.total_ms:.0f} ms | "
          f"{trace.tokens.get('input','?')} in / {trace.tokens.get('output','?')} out tokens ---")


if __name__ == "__main__":
    main()