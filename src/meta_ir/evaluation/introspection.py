
"""
META-IR Introspection Evaluation.

Evaluates structured introspection quality and
structural completeness.

A model-based judge must be supplied for the actual
introspection-quality score.
"""

import math

from dataclasses import dataclass
from typing import Callable, Sequence

from meta_ir.reasoning.introspection import (
    IntrospectionTrajectory,
    ReasoningStep,
)


IntrospectionJudge = Callable[
    [str, Sequence[ReasoningStep]],
    float,
]


@dataclass(frozen=True)
class IntrospectionResult:

    quality_score: float

    structural_coverage: float

    missing_stages: tuple[str, ...]


class IntrospectionEvaluator:
    """Structured introspection evaluator."""

    def __init__(
        self,
        judge: IntrospectionJudge,
        required_stages: Sequence[str] | None = None,
    ):

        self.judge = judge

        self.required_stages = tuple(
            required_stages
            or IntrospectionTrajectory.REQUIRED_STAGES
        )

    def structural_coverage(
        self,
        steps: Sequence[ReasoningStep],
    ) -> tuple[float, tuple[str, ...]]:

        completed = {
            step.stage for step in steps
        }

        missing = tuple(
            stage
            for stage in self.required_stages
            if stage not in completed
        )

        if not self.required_stages:
            return 1.0, ()

        coverage = (
            len(self.required_stages) - len(missing)
        ) / len(self.required_stages)

        return coverage, missing

    def evaluate(
        self,
        prompt: str,
        steps: Sequence[ReasoningStep],
    ) -> IntrospectionResult:

        quality = self.judge(prompt, steps)

        if not math.isfinite(quality):
            raise ValueError(
                "Introspection score must be finite."
            )

        if not -1.0 <= quality <= 1.0:
            raise ValueError(
                "Introspection score must belong to [-1, 1]."
            )

        coverage, missing = self.structural_coverage(
            steps
        )

        return IntrospectionResult(
            quality_score=quality,
            structural_coverage=coverage,
            missing_stages=missing,
        )
