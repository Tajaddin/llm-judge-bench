# llm-judge-bench

> LLM-as-judge framework with **calibrated rubrics** and **three named bias diagnostics**: position bias (verdict flips on candidate swap), length bias (verdict flips when the loser is padded), and inter-judge agreement (Cohen's κ). Designed so an interview-ready demo runs in <1 second with mocked judges and any number of real LLMs works the same way.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-9%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero demo

Three known-pathological mock judges run against the built-in 30-pair eval set. Each is designed to expose a different bias the framework should catch. Result table from `python bench/demo_mock.py`:

| Judge | Position-bias flip rate | "Picks first" rate | Length-bias flip rate | What it does |
|---|---:|---:|---:|---|
| `first-bias` (always picks A) | **100.0%** | **100.0%** | 0.0% | Picks the first candidate every time. Framework correctly identifies a 100% position bias. |
| `length` (picks the longer one) | 16.7% | 16.7% | **83.3%** | Picks whichever candidate is longer. Padding the loser flips 25/30 verdicts. |
| `content-keyword` | 66.7% | 66.7% | 0.0% | Picks based on domain keywords with "pick A on tie" fallback — caught as a milder position bias. |

Inter-judge agreement (Cohen's κ over the three pairs):

| Judge A | Judge B | Agreement rate | Cohen's κ |
|---|---|---:|---:|
| first-bias | length | 46.7% | 0.000 |
| first-bias | content-keyword | 83.3% | 0.000 |
| length | content-keyword | 63.3% | 0.298 |

Full output: [`bench/demo_results.json`](bench/demo_results.json).

The framework cleanly caught every pathology it should have. Run the same harness against real LLMs with `python bench/run_judges.py --judges llama-3.1-8b-instant llama-3.3-70b-versatile` once you have GROQ_API_KEY set.

## Why this exists

Every team that ships LLM features uses LLM-as-judge for some eval. The well-known failure modes — position bias, length bias, self-preference — are usually only addressed by "swap order and average" prompts, with no actual measurement. This framework gives you the measurement:

1. **Position bias** — for each pair, run the judge twice with A and B swapped. Any pair whose verdict changed counts as a flip. A truly impartial judge produces zero flips.
2. **Length bias** — for each pair where the judge picked a winner, pad the loser to match the winner's length and re-judge. Any flip toward the now-longer "loser" is a length-bias hit.
3. **Inter-judge agreement** — Cohen's κ between any two judges over the same pair list. κ = 1 means identical decisions, κ ≤ 0 means worse than chance.

The framework decouples judges (pluggable LLM backends) from rubrics (configurable system prompts) so swapping in your own grading criteria is one constructor parameter.

## Quickstart

```bash
pip install -e ".[groq,anthropic,dev]"
```

```python
from llm_judge_bench import (
    BiasAnalyzer, GroqJudgeBackend, Judge, JudgePair,
)
from llm_judge_bench.dataset import builtin_pairs

judges = [
    Judge(backend=GroqJudgeBackend(model="llama-3.1-8b-instant")),
    Judge(backend=GroqJudgeBackend(model="llama-3.3-70b-versatile")),
]

report = BiasAnalyzer(builtin_pairs()).run(judges)
print(report["position_bias"])
print(report["length_bias"])
print(report["agreements"])
```

Bring your own pairs:

```python
pairs = [
    JudgePair(
        item_id="my-1",
        question="Which response is more helpful?",
        candidate_a="Refer to the manual on page 14.",
        candidate_b="Sure! Page 14 of the manual covers this step-by-step.",
        gold_winner=None,
    ),
    # ...
]
report = BiasAnalyzer(pairs).run(judges)
```

Bring your own rubric:

```python
my_rubric = """
You are a clinical-grade reviewer. Reject any answer that gives medical advice
or names a drug not in the source passage. Reply VERDICT: A | B | TIE and one
short reason.
""".strip()

judge = Judge(backend=GroqJudgeBackend(model="llama-3.3-70b-versatile"), rubric=my_rubric)
```

## Built-in dataset

`llm_judge_bench.dataset.builtin_pairs()` returns 30 hand-curated common-knowledge pairs (capital of France, who wrote Hamlet, etc.) with a clear correct answer per pair. Every pair has `gold_winner` set so the framework can also report judge accuracy vs gold (in addition to the three bias diagnostics).

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
9 passed
```

Coverage:

* `_always_first_judge` triggers 100% position-bias flip rate (the framework caught it)
* `_content_judge` (honest) produces 0% position-bias flip rate
* `_length_judge` (picks the longer side) triggers 100% length-bias flip rate
* Cohen's κ math: perfect agreement = 1.0; total disagreement = ≤ 0; None labels dropped
* End-to-end run on the 3-pair subset returns the expected report shape
* The built-in dataset has 30 pairs, all with gold winners

## Project layout

```
.
├── src/llm_judge_bench/
│   ├── __init__.py
│   ├── llm.py          # Mock / Groq / Anthropic backends
│   ├── judge.py        # Judge wrapper + rubric + parse_verdict
│   ├── bias.py         # position / length / agreement measurement
│   └── dataset.py      # built-in 30-pair eval set
├── tests/              # 9 pytest cases
└── bench/
    ├── demo_mock.py        # mock-based demo (deterministic, <1s)
    ├── demo_results.json   # output of demo_mock.py
    └── run_judges.py       # real-judge benchmark via Groq
```

## Limitations

**The hero benchmark above uses mock judges.** Real-judge numbers from a live `bench/run_judges.py` run would land in `bench/judge_results.json`. During development of this repo the Groq free-tier daily token cap (500K/day, shared across earlier projects in the same session) prevented a full live run. The mock judges are deliberately pathological so the framework's response to each bias is auditable — they're not a replacement for live numbers, but they confirm the diagnostics fire correctly. The CLI is set up to produce identical-shape output once Groq budget is available again or you point it at Anthropic.

**Position-bias swap doubles judge calls.** Each pair becomes 2 LLM calls just for position measurement. On a 30-pair set with 2 judges that's 120 calls before length-bias even starts. For large benchmarks, run only one diagnostic at a time or sample pairs.

**Length-bias padding is structural.** The padding string is `" (additional context omitted)"` repeated until lengths match. A "smart" length-biased judge that responds to actual length-vs-content density wouldn't be caught by this padding. For deeper diagnostics, use the same content padded with paraphrased filler — out of scope for v0.1.

**Inter-judge agreement assumes the same pair set.** If two judges scored different subsets of pairs, the per-judge labels list will have `None` entries that get dropped pairwise. Cohen's κ is computed over only the pairs both judges decided on.

## License

MIT — see [LICENSE](LICENSE).
