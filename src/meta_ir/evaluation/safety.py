
"""
META-IR Safety Evaluation.

Evaluates safety scores using an externally supplied
safety judge.
"""

import math

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Sequence

from meta_ir.reasoning.introspection import ReasoningStep


SafetyJudge = Callable[
    [str, Sequence[ReasoningStep], str],
    float,
]


@dataclass(frozen=True)
class SafetySummary:

    total_samples: int

    mean_score: float | None

    safe_rate: float | None

    violation_rate: float | None


class SafetyEvaluator:
    """Safety score evaluator."""

    def __init__(
        self,
        judge: SafetyJudge,
        threshold: float = 0.0,
    ):

        self.judge = judge
        self.threshold = threshold

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
            raise ValueError("Safety score must be finite.")

        if not -1.0 <= value <= 1.0:
            raise ValueError(
                "Safety score must belong to [-1, 1]."
            )

        return value

    def summarize(
        self,
        scores: Sequence[float],
    ) -> SafetySummary:

        if not scores:
            return SafetySummary(
                total_samples=0,
                mean_score=None,
                safe_rate=None,
                violation_rate=None,
            )

        safe_count = sum(
            score >= self.threshold
            for score in scores
        )

        total = len(scores)

        return SafetySummary(
            total_samples=total,
            mean_score=mean(scores),
            safe_rate=safe_count / total,
            violation_rate=1.0 - safe_count / total,
        )
