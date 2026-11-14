
"""
SI-MCTS Search Tree Node.

Each node represents a partial introspective trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from meta_ir.reasoning.introspection import ReasoningStep


@dataclass
class SearchNode:

    state: tuple[ReasoningStep, ...]

    parent: Optional["SearchNode"] = None

    children: list["SearchNode"] = field(default_factory=list)

    visit_count: int = 0

    value_sum: float = 0.0

    safe_value_count: int = 0

    safety_score: float = 0.0

    pruned: bool = False

    expanded: bool = False

    @property
    def depth(self) -> int:
        return len(self.state)

    @property
    def mean_value(self) -> float:
        if self.safe_value_count == 0:
            return 0.0

        return self.value_sum / self.safe_value_count

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(
        self,
        child: "SearchNode",
    ) -> None:

        child.parent = self
        self.children.append(child)

    def update_visit(self) -> None:
        self.visit_count += 1

    def update_value(self, reward: float) -> None:

        self.value_sum += reward
        self.safe_value_count += 1
