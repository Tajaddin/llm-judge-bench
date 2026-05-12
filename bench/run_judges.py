"""Benchmark: run position-bias, length-bias, and pairwise agreement diagnostics
over the built-in 30-pair eval set with two real judges via Groq.

Default judges: llama-3.1-8b-instant + llama-3.3-70b-versatile (both Groq free).
Total LLM calls per judge ≈ 30 (position normal) + 30 (position swapped)
                          + ~30 (length-bias second pass on clean-decided pairs)
                          + 30 (agreement) ≈ 120 calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_judge_bench import BiasAnalyzer, GroqJudgeBackend, Judge
from llm_judge_bench.dataset import builtin_pairs


BENCH = Path(__file__).resolve().parent


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--judges", nargs="+", default=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--out", type=str, default=str(BENCH / "judge_results.json"))
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        return 2

    judges = [Judge(backend=GroqJudgeBackend(model=m)) for m in args.judges]
    pairs = builtin_pairs()[: args.limit]
    print(f"Running {len(judges)} judges × {len(pairs)} pairs...")

    report = BiasAnalyzer(pairs).run(judges)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
