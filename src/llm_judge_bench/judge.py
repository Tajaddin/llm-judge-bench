"""Judge + JudgePair + Verdict — the core of the framework.

A :class:`Judge` wraps a backend with a fixed system prompt + rubric. The
output is parsed into one of three :class:`Verdict` values (``A``, ``B``,
``TIE``). The rubric is configurable so users can swap in their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from llm_judge_bench.llm import JudgeBackend


class Verdict(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"


@dataclass
class JudgePair:
    """One (question, candidate_a, candidate_b) row.

    ``gold_winner`` is optional; supply it when you have a ground-truth
    annotation so the framework can also report accuracy vs gold.
    """

    item_id: str
    question: str
    candidate_a: str
    candidate_b: str
    gold_winner: Verdict | None = None
    extra: dict | None = None


DEFAULT_RUBRIC = (
    "You are an impartial judge. Compare two candidate answers to the user's question. "
    "Decide which answer is more correct and helpful. If they are equally good, you may "
    "answer TIE — but tie should be your last resort, not your default. "
    "Reply with EXACTLY one line of the form:\n"
    "VERDICT: <A or B or TIE>\n"
    "Followed by one short sentence of justification on a second line."
)


_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(A|B|TIE)\b", re.IGNORECASE)


def parse_verdict(raw: str) -> tuple[Verdict | None, str]:
    """Return ``(verdict, justification)``. ``verdict`` is None if unparseable."""
    if not raw:
        return None, ""
    m = _VERDICT_RE.search(raw)
    if not m:
        return None, raw.strip()
    v = Verdict(m.group(1).upper())
    # Justification is whatever comes after the VERDICT line.
    after = raw[m.end():].strip()
    return v, after


@dataclass
class Judge:
    """Wrap a backend with a rubric + a formatter for each pair."""

    backend: JudgeBackend
    rubric: str = DEFAULT_RUBRIC

    @property
    def model(self) -> str:
        return self.backend.model

    def render_prompt(self, pair: JudgePair, *, swap: bool = False) -> str:
        a, b = pair.candidate_a, pair.candidate_b
        if swap:
            a, b = b, a
        return (
            f"Question:\n{pair.question}\n\n"
            f"Candidate A:\n{a}\n\n"
            f"Candidate B:\n{b}\n\n"
            "Pick the better candidate. Output the VERDICT line then a one-sentence reason."
        )

    def judge(self, pair: JudgePair, *, swap: bool = False) -> tuple[Verdict | None, str, str]:
        """Return ``(verdict_in_swapped_frame, justification, raw)``.

        Note: when ``swap=True``, ``Verdict.A`` in the response refers to the
        ORIGINAL candidate_b. Callers must un-swap when comparing to gold.
        """
        prompt = self.render_prompt(pair, swap=swap)
        raw = self.backend.complete(prompt, system=self.rubric)
        verdict, justification = parse_verdict(raw)
        return verdict, justification, raw

    @staticmethod
    def unswap(verdict: Verdict | None) -> Verdict | None:
        """Translate a verdict from the swapped frame back to the original."""
        if verdict is None or verdict == Verdict.TIE:
            return verdict
        return Verdict.A if verdict == Verdict.B else Verdict.B
