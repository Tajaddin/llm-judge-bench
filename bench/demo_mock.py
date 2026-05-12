"""Mock-based demonstration of the bias analyzer.

Three synthetic judges:
* ``first-bias-judge``  — always picks Candidate A
* ``length-judge``      — picks whichever candidate is longer
* ``content-judge``     — picks whichever candidate contains the word "good"

Run against the built-in 30-pair dataset to show what the analyzer outputs
for three known-pathological judges. This script is deterministic, runs in
under a second, and is intended as the README's "what does the output look
like?" example.

For a real-judge run, use ``bench/run_judges.py`` with a Groq or Anthropic
API key.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_judge_bench import BiasAnalyzer, Judge, MockJudgeBackend
from llm_judge_bench.dataset import builtin_pairs

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "demo_results.json"


def _first_bias_judge() -> Judge:
    return Judge(backend=MockJudgeBackend(model="first-bias", handler=lambda p, s: "VERDICT: A\npicks first"))


def _length_judge() -> Judge:
    def handler(prompt: str, system: str | None) -> str:
        m_a = re.search(r"Candidate A:\n([\s\S]*?)\n\nCandidate B:", prompt)
        m_b = re.search(r"Candidate B:\n([\s\S]*?)\n\nPick the better", prompt)
        if not m_a or not m_b:
            return "VERDICT: TIE\nunparseable."
        return "VERDICT: A\nlonger\n" if len(m_a.group(1)) >= len(m_b.group(1)) else "VERDICT: B\nlonger\n"

    return Judge(backend=MockJudgeBackend(model="length", handler=handler))


def _content_judge() -> Judge:
    """Crude oracle: picks the candidate whose first letter matches the gold winner."""

    def handler(prompt: str, system: str | None) -> str:
        # The gold winner isn't in the prompt; this judge fakes "correct" behavior
        # by checking which candidate contains domain-aware keywords like "Paris"
        # for capital questions. For demo purposes we just always answer A when the
        # text "Pacific" or "Shakespeare" appears in candidate A.
        m_a = re.search(r"Candidate A:\n([\s\S]*?)\n\nCandidate B:", prompt) or re.search(r"Candidate A:\n([\s\S]*)", prompt)
        m_b = re.search(r"Candidate B:\n([\s\S]*?)\n\nPick the better", prompt) or re.search(r"Candidate B:\n([\s\S]*)", prompt)
        if not m_a or not m_b:
            return "VERDICT: TIE"
        cand_a = (m_a.group(1) or "").lower()
        cand_b = (m_b.group(1) or "").lower()
        score_a = sum(1 for kw in ("paris", "shakespeare", "pacific", "wrote", "is a star", "carbon dioxide", "leonardo", "1969", "einstein", "blue whale") if kw in cand_a)
        score_b = sum(1 for kw in ("paris", "shakespeare", "pacific", "wrote", "is a star", "carbon dioxide", "leonardo", "1969", "einstein", "blue whale") if kw in cand_b)
        if score_a > score_b:
            return "VERDICT: A\nkeyword."
        if score_b > score_a:
            return "VERDICT: B\nkeyword."
        return "VERDICT: A\nfallback first."

    return Judge(backend=MockJudgeBackend(model="content-keyword", handler=handler))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    pairs = builtin_pairs()
    judges = [_first_bias_judge(), _length_judge(), _content_judge()]
    report = BiasAnalyzer(pairs).run(judges)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
