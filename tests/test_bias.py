"""Bias diagnostics — synthetic judges expose each pathology cleanly."""

from __future__ import annotations

import pytest

from llm_judge_bench import (
    BiasAnalyzer,
    Judge,
    JudgePair,
    MockJudgeBackend,
    Verdict,
)
from llm_judge_bench.bias import cohens_kappa, measure_agreement, measure_length_bias, measure_position_bias
from llm_judge_bench.dataset import builtin_pairs


def _always_first_judge() -> Judge:
    return Judge(backend=MockJudgeBackend(handler=lambda p, s: "VERDICT: A\nfirst."))


def _content_judge() -> Judge:
    """Picks whichever side contains 'good'."""

    def handler(prompt: str, system: str | None) -> str:
        idx_a = prompt.find("Candidate A:")
        idx_b = prompt.find("Candidate B:")
        idx_end = prompt.find("Pick the better")
        a_block = prompt[idx_a:idx_b]
        b_block = prompt[idx_b:idx_end if idx_end != -1 else len(prompt)]
        if "good" in a_block.lower() and "good" not in b_block.lower():
            return "VERDICT: A\nA is good."
        if "good" in b_block.lower() and "good" not in a_block.lower():
            return "VERDICT: B\nB is good."
        return "VERDICT: TIE\nequivalent."

    return Judge(backend=MockJudgeBackend(model="content", handler=handler))


def _length_judge() -> Judge:
    """Picks whichever side is longer by comparing only the candidate strings."""

    import re

    def handler(prompt: str, system: str | None) -> str:
        m_a = re.search(r"Candidate A:\n([\s\S]*?)\n\nCandidate B:", prompt)
        m_b = re.search(r"Candidate B:\n([\s\S]*?)\n\nPick the better", prompt)
        if not m_a or not m_b:
            return "VERDICT: TIE\nunparseable."
        return "VERDICT: A\nlonger.\n" if len(m_a.group(1)) >= len(m_b.group(1)) else "VERDICT: B\nlonger.\n"

    return Judge(backend=MockJudgeBackend(model="length", handler=handler))


def test_first_position_judge_is_caught() -> None:
    judge = _always_first_judge()
    pairs = [JudgePair(item_id=str(i), question="?", candidate_a="x", candidate_b="y") for i in range(10)]
    report = measure_position_bias(judge, pairs)
    # On EVERY pair, the judge picks A in both runs → flip on every pair.
    assert report.flip_rate == 1.0
    assert report.first_lean_rate == 1.0


def test_honest_content_judge_has_zero_position_bias() -> None:
    judge = _content_judge()
    pairs = [
        JudgePair(item_id="1", question="?", candidate_a="this is good", candidate_b="this is bad"),
        JudgePair(item_id="2", question="?", candidate_a="this is bad", candidate_b="this is good"),
    ]
    report = measure_position_bias(judge, pairs)
    assert report.flip_rate == 0.0


def test_length_judge_caught_by_length_bias() -> None:
    judge = _length_judge()
    pairs = [
        # clean verdict will pick whichever is longer; padding flips it.
        JudgePair(item_id="1", question="?", candidate_a="aaaaaaaaaa long", candidate_b="short"),
        JudgePair(item_id="2", question="?", candidate_a="short", candidate_b="aaaaaaaaaa long"),
    ]
    report = measure_length_bias(judge, pairs)
    assert report.flip_rate == 1.0  # padding the loser to match length flips the verdict every time


def test_cohens_kappa_perfect_agreement_is_one() -> None:
    labels = [Verdict.A, Verdict.B, Verdict.A, Verdict.TIE]
    k, n, na = cohens_kappa(labels, labels)
    assert k == 1.0
    assert n == 4
    assert na == 4


def test_cohens_kappa_total_disagreement_is_negative() -> None:
    a = [Verdict.A, Verdict.A, Verdict.A, Verdict.A]
    b = [Verdict.B, Verdict.B, Verdict.B, Verdict.B]
    k, n, na = cohens_kappa(a, b)
    assert na == 0
    assert k <= 0.0


def test_agreement_drops_unparseable_labels() -> None:
    a = [Verdict.A, None, Verdict.B]
    b = [Verdict.A, Verdict.B, Verdict.B]
    k, n, na = cohens_kappa(a, b)
    assert n == 2  # the None entry is dropped
    assert na == 2


def test_bias_analyzer_runs_end_to_end_with_mocks() -> None:
    pairs = builtin_pairs()[:3]
    judges = [_content_judge(), _length_judge()]
    report = BiasAnalyzer(pairs).run(judges)
    assert "position_bias" in report
    assert "length_bias" in report
    assert "agreements" in report
    assert len(report["position_bias"]) == 2
    assert len(report["agreements"]) == 1  # one pair of judges → one agreement entry


def test_builtin_dataset_has_30_pairs() -> None:
    pairs = builtin_pairs()
    assert len(pairs) == 30
    assert all(p.gold_winner in (Verdict.A, Verdict.B) for p in pairs)


def test_judge_unswap_round_trip() -> None:
    assert Judge.unswap(Verdict.A) == Verdict.B
    assert Judge.unswap(Verdict.B) == Verdict.A
    assert Judge.unswap(Verdict.TIE) == Verdict.TIE
    assert Judge.unswap(None) is None
