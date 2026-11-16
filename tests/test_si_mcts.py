
"""
Tests for META-IR Safety-Informed Monte Carlo Tree Search.
"""

import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from meta_ir.reasoning.introspection import (
    ReasoningStep,
)

from meta_ir.si_mcts.search import SI_MCTS

from meta_ir.rewards.reward_model import (
    RewardModel,
)


# ============================================================
# 1. Deterministic Test Functions
# ============================================================

def helpfulness_function(
    prompt,
    steps,
    response="",
):
    return 0.8


def introspection_function(
    prompt,
    steps,
    response="",
):
    return 0.7


def make_reward_model(safety_function):

    return RewardModel(
        safety_function=safety_function,
        helpfulness_function=helpfulness_function,
        introspection_function=introspection_function,
        lambda_weight=0.5,
        safety_threshold=0.0,
    )


# ============================================================
# 2. Candidate Generator
# ============================================================

def safe_candidate_generator(
    prompt,
    previous_steps,
    num_candidates,
):

    stage = (
        "problem_framing"
        if not previous_steps
        else "safe_trajectory_planning"
    )

    return [
        ReasoningStep(
            stage=stage,
            title=f"Safe candidate {index}",
            explanation="A policy-compliant reasoning step.",
        )
        for index in range(num_candidates)
    ]


# ============================================================
# 3. Search Initialization
# ============================================================

def test_si_mcts_initialization():

    reward_model = make_reward_model(
        lambda prompt, steps: 1.0
    )

    engine = SI_MCTS(
        candidate_function=safe_candidate_generator,
        safety_function=lambda prompt, steps: 1.0,
        reward_model=reward_model,
        max_depth=3,
        num_simulations=5,
        branching_factor=2,
        exploration_coefficient=1.414,
        safety_threshold=0.0,
        seed=42,
    )

    assert engine is not None


# ============================================================
# 4. Search Execution
# ============================================================

def test_si_mcts_search_returns_result():

    reward_model = make_reward_model(
        lambda prompt, steps: 1.0
    )

    engine = SI_MCTS(
        candidate_function=safe_candidate_generator,
        safety_function=lambda prompt, steps: 1.0,
        reward_model=reward_model,
        max_depth=2,
        num_simulations=4,
        branching_factor=2,
        exploration_coefficient=1.414,
        safety_threshold=0.0,
        seed=42,
    )

    result = engine.search(
        "Explain trustworthy AI."
    )

    assert result is not None

    assert result.safe is True

    assert result.selected_trajectory

    assert result.simulations > 0


# ============================================================
# 5. Safety-Pruning Tests
# ============================================================

def mixed_candidate_generator(
    prompt,
    previous_steps,
    num_candidates,
):

    return [
        ReasoningStep(
            stage="safe_branch",
            title="Safe candidate",
            explanation="Policy-compliant candidate.",
        ),

        ReasoningStep(
            stage="unsafe_branch",
            title="Unsafe candidate",
            explanation="Policy-violating candidate.",
        ),
    ][:num_candidates]


def test_si_mcts_prunes_unsafe_candidates():

    def safety_function(prompt, steps):

        if any(
            step.stage == "unsafe_branch"
            for step in steps
        ):
            return -1.0

        return 1.0

    reward_model = make_reward_model(
        safety_function
    )

    engine = SI_MCTS(
        candidate_function=mixed_candidate_generator,
        safety_function=safety_function,
        reward_model=reward_model,
        max_depth=2,
        num_simulations=8,
        branching_factor=2,
        exploration_coefficient=1.414,
        safety_threshold=0.0,
        seed=42,
    )

    result = engine.search(
        "Provide a safe explanation of AI security."
    )

    assert result.safe is True

    assert all(
        step.stage != "unsafe_branch"
        for step in result.selected_trajectory
    )


# ============================================================
# 6. Safety Threshold Tests
# ============================================================

def test_si_mcts_rejects_unsafe_search():

    def unsafe_candidates(
        prompt,
        previous_steps,
        num_candidates,
    ):

        return [
            ReasoningStep(
                stage="unsafe_branch",
                title="Unsafe",
                explanation="Unsafe candidate.",
            )
        ]

    def unsafe_safety_function(prompt, steps):
        return -1.0

    reward_model = make_reward_model(
        unsafe_safety_function
    )

    engine = SI_MCTS(
        candidate_function=unsafe_candidates,
        safety_function=unsafe_safety_function,
        reward_model=reward_model,
        max_depth=2,
        num_simulations=4,
        branching_factor=1,
        exploration_coefficient=1.414,
        safety_threshold=0.0,
        seed=42,
    )

    result = engine.search(
        "Test unsafe candidate rejection."
    )

    assert result.safe is False
