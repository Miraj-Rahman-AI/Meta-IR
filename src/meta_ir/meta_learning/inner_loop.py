
"""
META-IR Inner Loop.

Uses SI-MCTS to refine an individual introspective
reasoning trajectory.
"""

from dataclasses import dataclass

from meta_ir.reasoning.introspection import (
    IntrospectionTrajectory,
)

from meta_ir.si_mcts.search import SI_MCTS, SearchResult


@dataclass
class InnerLoopResult:

    trajectory: IntrospectionTrajectory

    search_result: SearchResult

    safe: bool

    reward: float | None


class InnerLoop:
    """Per-prompt SI-MCTS trajectory refinement."""

    def __init__(self, search_engine: SI_MCTS):

        self.search_engine = search_engine

    def run(self, prompt: str) -> InnerLoopResult:

        result = self.search_engine.search(prompt)

        trajectory = IntrospectionTrajectory(
            prompt=prompt,
            steps=list(result.selected_trajectory),
        )

        reward = (
            result.reward.total
            if result.reward is not None
            else None
        )

        return InnerLoopResult(
            trajectory=trajectory,
            search_result=result,
            safe=result.safe,
            reward=reward,
        )
