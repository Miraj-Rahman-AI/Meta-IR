
"""
Safety-Guided Pruning.

Unsafe nodes and their descendants are excluded
from further exploration.
"""

from meta_ir.si_mcts.node import SearchNode


def is_safety_feasible(
    safety_score: float,
    threshold: float = 0.0,
) -> bool:

    return safety_score >= threshold


def prune_node(
    node: SearchNode,
    recursive: bool = True,
) -> None:
    """Permanently mark a node as pruned."""

    node.pruned = True

    if recursive:
        for child in node.children:
            prune_node(child, recursive=True)


def prune_unsafe_children(
    parent: SearchNode,
    threshold: float = 0.0,
) -> int:
    """Prune children whose safety scores are below threshold."""

    pruned_count = 0

    for child in parent.children:

        if child.safety_score < threshold:

            if not child.pruned:
                pruned_count += 1

            prune_node(child)

    return pruned_count
