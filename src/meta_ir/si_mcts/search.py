
"""
Safety-Informed Monte Carlo Tree Search.

Implements:
- UCB-based node selection.
- Candidate expansion.
- Safety-guided pruning.
- Safe trajectory rollout.
- Safe-only reward backpropagation.
"""

import math
import random

from dataclasses import dataclass
from typing import Callable, Optional

from meta_ir.reasoning.introspection import ReasoningStep
from meta_ir.si_mcts.node import SearchNode
from meta_ir.si_mcts.pruning import (
    is_safety_feasible,
    prune_node,
)
from meta_ir.rewards.reward_model import (
    RewardModel,
    RewardResult,
)


CandidateFunction = Callable[
    [str, tuple[ReasoningStep, ...], int],
    list[ReasoningStep],
]

SafetyFunction = Callable[
    [str, tuple[ReasoningStep, ...]],
    float,
]


@dataclass
class SearchResult:

    selected_trajectory: tuple[ReasoningStep, ...]

    reward: Optional[RewardResult]

    safe: bool

    simulations: int

    root: SearchNode


class SI_MCTS:
    """Safety-Informed Monte Carlo Tree Search."""

    def __init__(
        self,
        candidate_function: CandidateFunction,
        safety_function: SafetyFunction,
        reward_model: RewardModel,
        max_depth: int = 4,
        num_simulations: int = 16,
        branching_factor: int = 3,
        exploration_coefficient: float = 1.414,
        safety_threshold: float = 0.0,
        seed: int = 42,
    ):

        if max_depth < 1:
            raise ValueError("max_depth must be positive.")

        if num_simulations < 1:
            raise ValueError("num_simulations must be positive.")

        self.candidate_function = candidate_function
        self.safety_function = safety_function
        self.reward_model = reward_model

        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.branching_factor = branching_factor

        self.exploration_coefficient = exploration_coefficient
        self.safety_threshold = safety_threshold

        self.rng = random.Random(seed)

    # --------------------------------------------------------
    # UCB Selection
    # --------------------------------------------------------

    def _ucb_score(
        self,
        parent: SearchNode,
        child: SearchNode,
    ) -> float:

        if child.visit_count == 0:
            return float("inf")

        exploitation = child.mean_value

        exploration = (
            self.exploration_coefficient
            * math.sqrt(
                math.log(parent.visit_count + 1)
                / child.visit_count
            )
        )

        return exploitation + exploration

    def _select_child(
        self,
        node: SearchNode,
    ) -> Optional[SearchNode]:

        candidates = [
            child
            for child in node.children
            if not child.pruned
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda child: self._ucb_score(node, child),
        )

    # --------------------------------------------------------
    # Candidate Expansion
    # --------------------------------------------------------

    def _expand(
        self,
        prompt: str,
        node: SearchNode,
    ) -> None:

        if node.expanded:
            return

        candidates = self.candidate_function(
            prompt,
            node.state,
            self.branching_factor,
        )

        node.expanded = True

        for step in candidates:

            candidate_state = node.state + (step,)

            safety_score = self.safety_function(
                prompt,
                candidate_state,
            )

            if not math.isfinite(safety_score):
                raise ValueError("Invalid safety score.")

            child = SearchNode(
                state=candidate_state,
                safety_score=safety_score,
                pruned=not is_safety_feasible(
                    safety_score,
                    self.safety_threshold,
                ),
            )

            node.add_child(child)

    # --------------------------------------------------------
    # Rollout
    # --------------------------------------------------------

    def _rollout(
        self,
        prompt: str,
        initial_state: tuple[ReasoningStep, ...],
    ) -> tuple[tuple[ReasoningStep, ...], bool]:

        state = initial_state

        while len(state) < self.max_depth:

            candidates = self.candidate_function(
                prompt,
                state,
                self.branching_factor,
            )

            safe_candidates = []

            for step in candidates:

                candidate_state = state + (step,)

                score = self.safety_function(
                    prompt,
                    candidate_state,
                )

                if not math.isfinite(score):
                    raise ValueError("Invalid safety score.")

                if is_safety_feasible(
                    score,
                    self.safety_threshold,
                ):
                    safe_candidates.append((step, score))

            if not safe_candidates:
                return state, False

            # Select a safety-feasible candidate.
            # This is a rollout policy, not an additional
            # reward-based optimization step.
            step, _ = max(
                safe_candidates,
                key=lambda item: item[1],
            )

            state = state + (step,)

        final_score = self.safety_function(
            prompt,
            state,
        )

        return (
            state,
            is_safety_feasible(
                final_score,
                self.safety_threshold,
            ),
        )

    # --------------------------------------------------------
    # Reward Backpropagation
    # --------------------------------------------------------

    def _backpropagate(
        self,
        path: list[SearchNode],
        reward: Optional[float],
    ) -> None:

        for node in path:

            node.update_visit()

            # Unsafe rollouts never update node values.
            if reward is not None:
                node.update_value(reward)

    # --------------------------------------------------------
    # Main Search
    # --------------------------------------------------------

    def search(self, prompt: str) -> SearchResult:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        root = SearchNode(state=())

        best_trajectory = ()
        best_reward = None

        for _ in range(self.num_simulations):

            node = root
            path = [root]

            # Tree selection and expansion.
            while node.depth < self.max_depth:

                self._expand(prompt, node)

                next_node = self._select_child(node)

                if next_node is None:
                    break

                node = next_node
                path.append(node)

            # Complete a safe rollout.
            trajectory, safe = self._rollout(
                prompt,
                node.state,
            )

            reward_value = None

            if safe:

                reward = self.reward_model.evaluate(
                    prompt,
                    trajectory,
                )

                safe = reward.is_safe

                if safe:

                    reward_value = reward.total

                    if (
                        best_reward is None
                        or reward.total > best_reward.total
                    ):
                        best_reward = reward
                        best_trajectory = trajectory

            # Safe-only reward propagation.
            self._backpropagate(
                path,
                reward_value,
            )

        return SearchResult(
            selected_trajectory=best_trajectory,
            reward=best_reward,
            safe=best_reward is not None,
            simulations=self.num_simulations,
            root=root,
        )
