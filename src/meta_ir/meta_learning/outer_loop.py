
"""
META-IR Outer Loop.

Aggregates task-level rewards and delegates the
introspection-policy update to a model-specific updater.

The SI-MCTS search itself is not differentiated through.
"""

from dataclasses import dataclass
from statistics import mean
from typing import Protocol, Sequence

from meta_ir.jailbreak.simulation import (
    JailbreakSimulator,
    JailbreakTask,
)

from meta_ir.meta_learning.inner_loop import (
    InnerLoop,
    InnerLoopResult,
)


@dataclass(frozen=True)
class MetaTrainingExample:

    task_family: str

    prompt: str

    trajectory: tuple

    reward: float

    is_safe: bool


@dataclass
class MetaIterationResult:

    iteration: int

    examples: list[MetaTrainingExample]

    mean_task_reward: float | None

    safe_count: int

    total_count: int

    policy_updated: bool


class PolicyUpdater(Protocol):
    """
    Interface for the actual introspection-policy optimizer.

    Implement this interface using your trainable policy
    and the appropriate gradient-based learning procedure.
    """

    def update(
        self,
        examples: Sequence[MetaTrainingExample],
    ) -> None:
        ...


class OuterLoop:
    """Cross-task introspection-policy adaptation."""

    def __init__(
        self,
        simulator: JailbreakSimulator,
        inner_loop: InnerLoop,
        policy_updater: PolicyUpdater,
    ):

        self.simulator = simulator
        self.inner_loop = inner_loop
        self.policy_updater = policy_updater

        self.iteration = 0

    def run_iteration(
        self,
        batch_size: int,
    ) -> MetaIterationResult:

        self.iteration += 1

        tasks = self.simulator.sample_batch(batch_size)

        examples = []
        safe_rewards = []

        for task in tasks:

            result = self.inner_loop.run(task.prompt)

            if result.safe and result.reward is not None:

                example = MetaTrainingExample(
                    task_family=task.family,
                    prompt=task.prompt,
                    trajectory=tuple(result.trajectory.steps),
                    reward=result.reward,
                    is_safe=True,
                )

                examples.append(example)
                safe_rewards.append(result.reward)

        # Do not update using unsafe rollout rewards.
        policy_updated = bool(examples)

        if policy_updated:

            self.policy_updater.update(examples)

        return MetaIterationResult(
            iteration=self.iteration,
            examples=examples,
            mean_task_reward=(
                mean(safe_rewards)
                if safe_rewards
                else None
            ),
            safe_count=len(examples),
            total_count=len(tasks),
            policy_updated=policy_updated,
        )
