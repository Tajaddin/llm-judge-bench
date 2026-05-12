"""Three named bias / agreement diagnostics.

* ``PositionBiasReport``  — fraction of pairs whose verdict flips when the
  candidate order is swapped. A *truly* impartial judge would flip 0 pairs.
* ``LengthBiasReport``    — fraction of pairs whose verdict flips when the
  "losing" candidate is padded to match the winner's length. A judge that
  ignores length would flip 0 pairs.
* ``AgreementReport``     — Cohen's kappa between two judges' verdicts on
  the same pair list. κ = 1 means identical decisions; κ ≤ 0 means worse
  than chance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from llm_judge_bench.judge import Judge, JudgePair, Verdict


# ---------------------------------------------------------------------------
# Position bias
# ---------------------------------------------------------------------------


@dataclass
class PositionBiasResult:
    item_id: str
    verdict_normal: Verdict | None
    verdict_swapped_unswapped: Verdict | None
    flipped: bool
    leans_first: bool   # True iff the judge picked the *first* candidate in both runs


@dataclass
class PositionBiasReport:
    judge_model: str
    n: int
    flip_count: int
    flip_rate: float
    first_lean_count: int   # times judge picked the *first*-shown candidate in BOTH runs
    first_lean_rate: float
    rows: list[PositionBiasResult] = field(default_factory=list)


def measure_position_bias(judge: Judge, pairs: Iterable[JudgePair]) -> PositionBiasReport:
    rows: list[PositionBiasResult] = []
    flips = 0
    first_leans = 0
    n = 0
    for pair in pairs:
        n += 1
        v_normal, _, _ = judge.judge(pair, swap=False)
        v_swapped_raw, _, _ = judge.judge(pair, swap=True)
        v_swapped = Judge.unswap(v_swapped_raw)
        flipped = (
            v_normal is not None and v_swapped is not None and v_normal != v_swapped
        )
        leans_first = (
            v_normal == Verdict.A and v_swapped_raw == Verdict.A
        )
        if flipped:
            flips += 1
        if leans_first:
            first_leans += 1
        rows.append(
            PositionBiasResult(
                item_id=pair.item_id,
                verdict_normal=v_normal,
                verdict_swapped_unswapped=v_swapped,
                flipped=flipped,
                leans_first=leans_first,
            )
        )
    return PositionBiasReport(
        judge_model=judge.model,
        n=n,
        flip_count=flips,
        flip_rate=round(flips / max(n, 1), 4),
        first_lean_count=first_leans,
        first_lean_rate=round(first_leans / max(n, 1), 4),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Length bias
# ---------------------------------------------------------------------------


@dataclass
class LengthBiasResult:
    item_id: str
    verdict_clean: Verdict | None
    verdict_padded: Verdict | None
    flipped_toward_padded: bool


@dataclass
class LengthBiasReport:
    judge_model: str
    n: int
    flip_count: int
    flip_rate: float
    rows: list[LengthBiasResult] = field(default_factory=list)


def _pad_to_match(target: str, reference: str, *, pad: str = " (additional context omitted)") -> str:
    """Pad ``target`` until it's at least as long as ``reference``."""
    if len(target) >= len(reference):
        return target
    while len(target) < len(reference):
        target = target + pad
    return target


def measure_length_bias(judge: Judge, pairs: Iterable[JudgePair]) -> LengthBiasReport:
    """Pad the *clean-verdict loser* to match the winner's length and re-judge.

    A length-biased judge will flip toward the now-longer (originally loser)
    candidate. Pairs where the clean verdict was TIE or unparseable are skipped.
    """
    rows: list[LengthBiasResult] = []
    flips = 0
    n = 0
    for pair in pairs:
        v_clean, _, _ = judge.judge(pair, swap=False)
        if v_clean not in (Verdict.A, Verdict.B):
            continue
        n += 1
        loser = "B" if v_clean == Verdict.A else "A"
        padded_pair = JudgePair(
            item_id=pair.item_id,
            question=pair.question,
            candidate_a=_pad_to_match(pair.candidate_a, pair.candidate_b) if loser == "A" else pair.candidate_a,
            candidate_b=_pad_to_match(pair.candidate_b, pair.candidate_a) if loser == "B" else pair.candidate_b,
            gold_winner=pair.gold_winner,
        )
        v_padded, _, _ = judge.judge(padded_pair, swap=False)
        flipped_toward_loser = (
            v_padded is not None and v_padded != v_clean
        )
        if flipped_toward_loser:
            flips += 1
        rows.append(
            LengthBiasResult(
                item_id=pair.item_id,
                verdict_clean=v_clean,
                verdict_padded=v_padded,
                flipped_toward_padded=flipped_toward_loser,
            )
        )
    return LengthBiasReport(
        judge_model=judge.model,
        n=n,
        flip_count=flips,
        flip_rate=round(flips / max(n, 1), 4),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Agreement (Cohen's kappa)
# ---------------------------------------------------------------------------


@dataclass
class AgreementReport:
    judge_a_model: str
    judge_b_model: str
    n: int
    agree_count: int
    agree_rate: float
    cohens_kappa: float


def cohens_kappa(labels_a: list[Verdict | None], labels_b: list[Verdict | None]) -> tuple[float, int, int]:
    """Cohen's kappa over Verdict labels. None labels are dropped pairwise.

    Returns ``(kappa, n_compared, n_agree)``.
    """
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return 0.0, 0, 0
    n_agree = sum(1 for a, b in pairs if a == b)
    po = n_agree / n
    # Marginals.
    cats: list[Verdict] = [Verdict.A, Verdict.B, Verdict.TIE]
    pe = 0.0
    for c in cats:
        pa = sum(1 for a, _ in pairs if a == c) / n
        pb = sum(1 for _, b in pairs if b == c) / n
        pe += pa * pb
    if abs(1 - pe) < 1e-9:
        return 1.0, n, n_agree
    kappa = (po - pe) / (1 - pe)
    return float(kappa), n, n_agree


def measure_agreement(judge_a: Judge, judge_b: Judge, pairs: list[JudgePair]) -> AgreementReport:
    labels_a: list[Verdict | None] = []
    labels_b: list[Verdict | None] = []
    for pair in pairs:
        va, _, _ = judge_a.judge(pair, swap=False)
        vb, _, _ = judge_b.judge(pair, swap=False)
        labels_a.append(va)
        labels_b.append(vb)
    kappa, n, n_agree = cohens_kappa(labels_a, labels_b)
    return AgreementReport(
        judge_a_model=judge_a.model,
        judge_b_model=judge_b.model,
        n=n,
        agree_count=n_agree,
        agree_rate=round(n_agree / max(n, 1), 4),
        cohens_kappa=round(kappa, 4),
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class BiasAnalyzer:
    pairs: list[JudgePair]

    def run(self, judges: list[Judge]) -> dict:
        """Run all three diagnostics and return a serializable dict."""
        per_judge_position: list[PositionBiasReport] = []
        per_judge_length: list[LengthBiasReport] = []
        for j in judges:
            per_judge_position.append(measure_position_bias(j, self.pairs))
            per_judge_length.append(measure_length_bias(j, self.pairs))
        agreements: list[AgreementReport] = []
        for i in range(len(judges)):
            for k in range(i + 1, len(judges)):
                agreements.append(measure_agreement(judges[i], judges[k], self.pairs))
        return {
            "position_bias": [
                {
                    "judge": r.judge_model,
                    "n": r.n,
                    "flip_count": r.flip_count,
                    "flip_rate": r.flip_rate,
                    "first_lean_count": r.first_lean_count,
                    "first_lean_rate": r.first_lean_rate,
                }
                for r in per_judge_position
            ],
            "length_bias": [
                {
                    "judge": r.judge_model,
                    "n": r.n,
                    "flip_count": r.flip_count,
                    "flip_rate": r.flip_rate,
                }
                for r in per_judge_length
            ],
            "agreements": [
                {
                    "judge_a": a.judge_a_model,
                    "judge_b": a.judge_b_model,
                    "n": a.n,
                    "agree_rate": a.agree_rate,
                    "cohens_kappa": a.cohens_kappa,
                }
                for a in agreements
            ],
        }
