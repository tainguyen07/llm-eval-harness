"""Evaluation: regression diffing and gate enforcement."""

from llm_eval_harness.evaluation.diff import RegressionDiff, diff
from llm_eval_harness.evaluation.gates import GateReport, evaluate_gates

__all__ = [
    "GateReport",
    "RegressionDiff",
    "diff",
    "evaluate_gates",
]