
"""
META-IR Meta-Adaptive Response Generation.

Implements:
- Best-of-N candidate generation.
- Reward-guided beam expansion.
- Safety-feasibility filtering.
- Final response selection.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

from meta_ir.reasoning.introspection import ReasoningStep
from meta_ir.rewards.reward_model import (
    RewardModel,
    RewardResult,
)


@dataclass(frozen=True)
class CandidateResponse:

    steps: tuple[ReasoningStep, ...]

    response: str


@dataclass(frozen=True)
class GenerationResult:

    response: str

    steps: tuple[ReasoningStep, ...]

    reward: RewardResult | None

    used_fallback: bool


CandidateGenerator = Callable[
    [str, int],
    Sequence[CandidateResponse],
]

BeamExpander = Callable[
    [str, tuple[ReasoningStep, ...], int],
    Sequence[ReasoningStep],
]

ResponseGenerator = Callable[
    [str, tuple[ReasoningStep, ...]],
    str,
]


class AdaptiveGenerator:
    """Reward-guided safe response generation."""

    SAFE_FALLBACK = (
        "I cannot assist with that request. "
        "I can help with a safe alternative."
    )

    def __init__(
        self,
        reward_model: RewardModel,
        candidate_generator: CandidateGenerator,
        beam_expander: BeamExpander | None = None,
        response_generator: ResponseGenerator | None = None,
        beam_width: int = 3,
        max_depth: int = 4,
    ):

        self.reward_model = reward_model
        self.candidate_generator = candidate_generator
        self.beam_expander = beam_expander
        self.response_generator = response_generator

        self.beam_width = beam_width
        self.max_depth = max_depth

    # --------------------------------------------------------
    # Candidate Evaluation
    # --------------------------------------------------------

    def _evaluate(
        self,
        prompt: str,
        candidate: CandidateResponse,
    ) -> RewardResult:

        return self.reward_model.evaluate(
            prompt,
            candidate.steps,
            candidate.response,
        )

    def _select_best(
        self,
        prompt: str,
        candidates: Sequence[CandidateResponse],
    ) -> GenerationResult:

        safe_candidates = []

        for candidate in candidates:

            result = self._evaluate(prompt, candidate)

            if result.is_safe:

                safe_candidates.append(
                    (candidate, result)
                )

        if not safe_candidates:

            return GenerationResult(
                response=self.SAFE_FALLBACK,
                steps=(),
                reward=None,
                used_fallback=True,
            )

        candidate, reward = max(
            safe_candidates,
            key=lambda item: item[1].total,
        )

        return GenerationResult(
            response=candidate.response,
            steps=candidate.steps,
            reward=reward,
            used_fallback=False,
        )

    # --------------------------------------------------------
    # Best-of-N Sampling
    # --------------------------------------------------------

    def best_of_n(
        self,
        prompt: str,
        n: int = 5,
    ) -> GenerationResult:

        if n < 1:
            raise ValueError("n must be positive.")

        candidates = self.candidate_generator(
            prompt,
            n,
        )

        return self._select_best(
            prompt,
            candidates,
        )

    # --------------------------------------------------------
    # Reward-Guided Beam Search
    # --------------------------------------------------------

    def beam_search(
        self,
        prompt: str,
    ) -> GenerationResult:

        if (
            self.beam_expander is None
            or self.response_generator is None
        ):
            raise RuntimeError(
                "Beam expansion and response generation "
                "functions must be supplied."
            )

        beams: list[tuple[ReasoningStep, ...]] = [()]

        for _ in range(self.max_depth):

            expanded = []

            for prefix in beams:

                candidates = self.beam_expander(
                    prompt,
                    prefix,
                    self.beam_width,
                )

                for step in candidates:

                    new_state = prefix + (step,)

                    result = self.reward_model.evaluate(
                        prompt,
                        new_state,
                        "",
                    )

                    if result.is_safe:

                        expanded.append(
                            (new_state, result.total)
                        )

            if not expanded:
                break

            expanded.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            beams = [
                state
                for state, _ in expanded[:self.beam_width]
            ]

        candidates = []

        for state in beams:

            response = self.response_generator(
                prompt,
                state,
            )

            candidates.append(
                CandidateResponse(
                    steps=state,
                    response=response,
                )
            )

        return self._select_best(
            prompt,
            candidates,
        )

    # --------------------------------------------------------
    # Main Generation Interface
    # --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        method: str = "best_of_n",
        n: int = 5,
    ) -> GenerationResult:

        if method == "best_of_n":
            return self.best_of_n(prompt, n)

        if method == "beam_search":
            return self.beam_search(prompt)

        raise ValueError(
            f"Unsupported generation method: {method}"
        )
