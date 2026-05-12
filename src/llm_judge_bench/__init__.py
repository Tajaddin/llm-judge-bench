"""LLM-as-judge framework with bias diagnostics."""

from llm_judge_bench.bias import (
    AgreementReport,
    BiasAnalyzer,
    LengthBiasReport,
    PositionBiasReport,
)
from llm_judge_bench.judge import Judge, JudgePair, Verdict
from llm_judge_bench.llm import AnthropicJudgeBackend, GroqJudgeBackend, JudgeBackend, MockJudgeBackend

__version__ = "0.1.0"

__all__ = [
    "Judge",
    "JudgePair",
    "Verdict",
    "BiasAnalyzer",
    "AgreementReport",
    "PositionBiasReport",
    "LengthBiasReport",
    "JudgeBackend",
    "MockJudgeBackend",
    "GroqJudgeBackend",
    "AnthropicJudgeBackend",
]
