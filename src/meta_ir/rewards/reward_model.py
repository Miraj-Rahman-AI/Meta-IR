
"""
META-IR Composite Reward Optimization.

R(H,S,I) = F(S)H + S + lambda*I

The evaluator functions must be connected to actual
safety, helpfulness, and introspection judge models.
"""

import math

from dataclasses import dataclass
from typing import Callable, Sequence

from meta_ir.reasoning.introspection import ReasoningStep


ScoreFunction = Callable[
    [str, Sequence[ReasoningStep], str],
    float,
]


@dataclass(frozen=True)
class RewardComponents:

    helpfulness: float
    safety: float
    introspection: float


@dataclass(frozen=True)
class RewardResult:

    components: RewardComponents

    total: float

    is_safe: bool


class CompositeReward:
    """Calculate META-IR composite reward."""

    def __init__(
        self,
        lambda_weight: float,
        safety_threshold: float = 0.0,
    ):

        if lambda_weight <= 0:
            raise ValueError("lambda_weight must be positive.")

        self.lambda_weight = lambda_weight
        self.safety_threshold = safety_threshold

    @staticmethod
    def safety_modulation(safety: float) -> float:

        return 0.0 if safety < 0.0 else 1.0

    def calculate(
        self,
        components: RewardComponents,
    ) -> RewardResult:

        scores = (
            components.helpfulness,
            components.safety,
            components.introspection,
        )

        for score in scores:

            if not math.isfinite(score):
                raise ValueError("Reward score must be finite.")

            if not -1.0 <= score <= 1.0:
                raise ValueError(
                    "Reward scores must belong to [-1, 1]."
                )

        modulation = self.safety_modulation(
            components.safety
        )

        total = (
            modulation * components.helpfulness
            + components.safety
            + self.lambda_weight * components.introspection
        )

        return RewardResult(
            components=components,
            total=total,
            is_safe=(
                components.safety >= self.safety_threshold
            ),
        )


class RewardModel:
    """
    Model-independent reward evaluator.

    Three callable evaluators are required:
    - safety evaluator
    - helpfulness evaluator
    - introspection evaluator
    """

    def __init__(
        self,
        safety_function: ScoreFunction,
        helpfulness_function: ScoreFunction,
        introspection_function: ScoreFunction,
        lambda_weight: float,
        safety_threshold: float = 0.0,
    ):

        self.safety_function = safety_function
        self.helpfulness_function = helpfulness_function
        self.introspection_function = introspection_function

        self.composite_reward = CompositeReward(
            lambda_weight=lambda_weight,
            safety_threshold=safety_threshold,
        )

    def evaluate(
        self,
        prompt: str,
        steps: Sequence[ReasoningStep],
        final_response: str = "",
    ) -> RewardResult:

        safety = self.safety_function(
            prompt, steps, final_response
        )

        helpfulness = self.helpfulness_function(
            prompt, steps, final_response
        )

        introspection = self.introspection_function(
            prompt, steps, final_response
        )

        components = RewardComponents(
            helpfulness=helpfulness,
            safety=safety,
            introspection=introspection,
        )

        return self.composite_reward.calculate(components)
