
"""
Structured Introspective Reasoning.

Defines the representation of introspective trajectories
and an interface for generating candidate reasoning steps.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True)
class ReasoningStep:
    """One structured introspective reasoning step."""

    stage: str
    title: str
    explanation: str

    def __post_init__(self):
        if not self.stage.strip():
            raise ValueError("Reasoning stage cannot be empty.")

        if not self.title.strip():
            raise ValueError("Reasoning title cannot be empty.")

        if not self.explanation.strip():
            raise ValueError("Reasoning explanation cannot be empty.")

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "title": self.title,
            "explanation": self.explanation,
        }


@dataclass
class IntrospectionTrajectory:
    """Complete structured introspective trajectory."""

    prompt: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_response: str = ""

    REQUIRED_STAGES = (
        "problem_framing",
        "hidden_intent_detection",
        "risk_policy_analysis",
        "safe_trajectory_planning",
    )

    @property
    def is_complete(self) -> bool:
        stages = {step.stage for step in self.steps}
        return all(stage in stages for stage in self.REQUIRED_STAGES)

    def add_step(self, step: ReasoningStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "steps": [step.to_dict() for step in self.steps],
            "final_response": self.final_response,
            "is_complete": self.is_complete,
        }


CandidateFunction = Callable[
    [str, Sequence[ReasoningStep], int],
    Sequence[ReasoningStep],
]


class StructuredIntrospector:
    """
    Model-independent introspection controller.

    candidate_function must be connected to the actual
    LLM generation implementation.
    """

    def __init__(
        self,
        candidate_function: CandidateFunction,
    ):
        self.candidate_function = candidate_function

    def generate_candidates(
        self,
        prompt: str,
        previous_steps: Sequence[ReasoningStep] = (),
        num_candidates: int = 3,
    ) -> list[ReasoningStep]:

        if not prompt.strip():
            raise ValueError("Input prompt cannot be empty.")

        if num_candidates < 1:
            raise ValueError("num_candidates must be positive.")

        candidates = self.candidate_function(
            prompt,
            previous_steps,
            num_candidates,
        )

        return list(candidates)

    def generate_initial_trajectory(
        self,
        prompt: str,
        num_candidates: int = 1,
    ) -> IntrospectionTrajectory:

        steps = self.generate_candidates(
            prompt=prompt,
            previous_steps=(),
            num_candidates=num_candidates,
        )

        return IntrospectionTrajectory(
            prompt=prompt,
            steps=steps,
        )
