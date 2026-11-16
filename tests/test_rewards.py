
"""
Tests for META-IR Composite Reward Model.
"""

import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from meta_ir.reasoning.introspection import (
    ReasoningStep,
)

from meta_ir.rewards.reward_model import (
    RewardModel,
)


# ============================================================
# 1. Test Data
# ============================================================

PROMPT = "Explain trustworthy AI."

STEPS = [
    ReasoningStep(
        stage="problem_framing",
        title="Task identification",
        explanation="Identify the task.",
    )
]


def make_reward_model(
    helpfulness_score,
    safety_score,
    introspection_score,
    lambda_weight=0.5,
    safety_threshold=0.0,
):

    return RewardModel(
        safety_function=lambda prompt, steps, response="": (
            safety_score
        ),

        helpfulness_function=lambda prompt, steps, response="": (
            helpfulness_score
        ),

        introspection_function=lambda prompt, steps, response="": (
            introspection_score
        ),

        lambda_weight=lambda_weight,

        safety_threshold=safety_threshold,
    )


# ============================================================
# 2. Reward Calculation Tests
# ============================================================

def test_positive_safety_reward():

    model = make_reward_model(
        helpfulness_score=0.8,
        safety_score=0.6,
        introspection_score=0.4,
        lambda_weight=0.5,
    )

    result = model.evaluate(
        PROMPT,
        STEPS,
        "A helpful and safe answer.",
    )

    expected = (
        0.8 + 0.6 + (0.5 * 0.4)
    )

    assert result.total == pytest.approx(
        expected
    )

    assert result.is_safe is True


def test_negative_safety_suppresses_helpfulness():

    model = make_reward_model(
        helpfulness_score=0.8,
        safety_score=-0.5,
        introspection_score=0.4,
        lambda_weight=0.5,
    )

    result = model.evaluate(
        PROMPT,
        STEPS,
        "Unsafe answer.",
    )

    expected = (
        0.0 + (-0.5) + (0.5 * 0.4)
    )

    assert result.total == pytest.approx(
        expected
    )

    assert result.is_safe is False


def test_zero_safety_score():

    model = make_reward_model(
        helpfulness_score=0.8,
        safety_score=0.0,
        introspection_score=0.4,
        lambda_weight=0.5,
    )

    result = model.evaluate(
        PROMPT,
        STEPS,
        "Boundary case.",
    )

    expected = (
        0.8 + 0.0 + (0.5 * 0.4)
    )

    assert result.total == pytest.approx(
        expected
    )

    assert result.is_safe is True


# ============================================================
# 3. Reward Components
# ============================================================

def test_reward_components_are_available():

    model = make_reward_model(
        helpfulness_score=0.7,
        safety_score=0.9,
        introspection_score=0.6,
    )

    result = model.evaluate(
        PROMPT,
        STEPS,
        "Safe response.",
    )

    assert result.components.helpfulness == pytest.approx(
        0.7
    )

    assert result.components.safety == pytest.approx(
        0.9
    )

    assert result.components.introspection == pytest.approx(
        0.6
    )


# ============================================================
# 4. Safety Threshold
# ============================================================

def test_custom_safety_threshold():

    model = make_reward_model(
        helpfulness_score=0.8,
        safety_score=0.4,
        introspection_score=0.7,
        safety_threshold=0.5,
    )

    result = model.evaluate(
        PROMPT,
        STEPS,
        "Response below safety threshold.",
    )

    assert result.is_safe is False


def test_safety_above_custom_threshold():

    model = make_reward_model(
        helpfulness_score=0.8,
        safety_score=0.8,
        introspection_score=0.7,
        safety_threshold=0.5,
    )

    result = model.evaluate(
        PROMPT,
        STEPS,
        "Response above safety threshold.",
    )

    assert result.is_safe is True


# ============================================================
# 5. Invalid Configuration
# ============================================================

def test_reward_model_rejects_missing_judges():

    with pytest.raises((TypeError, ValueError)):

        RewardModel(
            safety_function=None,
            helpfulness_function=None,
            introspection_function=None,
            lambda_weight=0.5,
        )
