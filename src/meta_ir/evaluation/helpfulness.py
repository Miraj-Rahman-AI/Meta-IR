
"""
META-IR Helpfulness Evaluation.

Measures response usefulness using an external
helpfulness judge.
"""

import math

from statistics import mean
from typing import Callable, Sequence

from meta_ir.reasoning.introspection import ReasoningStep


HelpfulnessJudge = Callable[
    [str, Sequence[ReasoningStep], str],
    float,
]


class HelpfulnessEvaluator:
    """Helpfulness score evaluator."""

    def __init__(
        self,
        judge: HelpfulnessJudge,
    ):

        self.judge = judge

    def score(
        self,
        prompt: str,
        steps: Sequence[ReasoningStep],
        response: str = "",
    ) -> float:

        value = self.judge(
            prompt,
            steps,
            response,
        )

        if not math.isfinite(value):
            raise ValueError(
                "Helpfulness score must be finite."
            )

        if not -1.0 <= value <= 1.0:
            raise ValueError(
                "Helpfulness score must belong to [-1, 1]."
            )

        return value

    @staticmethod
    def summarize(
        scores: Sequence[float],
    ) -> dict:

        if not scores:
            return {
                "total_samples": 0,
                "mean_helpfulness": None,
            }

        return {
            "total_samples": len(scores),
            "mean_helpfulness": mean(scores),
        }
