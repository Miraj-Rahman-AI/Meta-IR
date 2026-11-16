
"""
Tests for META-IR Structured Introspection.
"""

import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from meta_ir.reasoning.introspection import (
    ReasoningStep,
    IntrospectionTrajectory,
    StructuredIntrospector,
)


# ============================================================
# 1. ReasoningStep Tests
# ============================================================

def test_reasoning_step_initialization():

    step = ReasoningStep(
        stage="problem_framing",
        title="Understand the task",
        explanation="Identify the user's main objective.",
    )

    assert step.stage == "problem_framing"

    assert step.title == "Understand the task"

    assert isinstance(step.explanation, str)


def test_reasoning_step_serialization():

    step = ReasoningStep(
        stage="risk_policy_analysis",
        title="Risk assessment",
        explanation="Assess relevant safety constraints.",
    )

    data = step.to_dict()

    assert data["stage"] == "risk_policy_analysis"

    assert data["title"] == "Risk assessment"

    assert data["explanation"] == (
        "Assess relevant safety constraints."
    )


# ============================================================
# 2. IntrospectionTrajectory Tests
# ============================================================

def test_trajectory_initialization():

    trajectory = IntrospectionTrajectory(
        prompt="Explain trustworthy AI."
    )

    assert trajectory.prompt == (
        "Explain trustworthy AI."
    )

    assert trajectory.steps == []


def test_add_reasoning_step():

    trajectory = IntrospectionTrajectory(
        prompt="Explain AI safety."
    )

    step = ReasoningStep(
        stage="problem_framing",
        title="Task identification",
        explanation="Identify the requested explanation.",
    )

    trajectory.add_step(step)

    assert len(trajectory.steps) == 1

    assert trajectory.steps[0] == step


def test_trajectory_serialization():

    step = ReasoningStep(
        stage="safe_trajectory_planning",
        title="Safe response",
        explanation="Prepare a safe response.",
    )

    trajectory = IntrospectionTrajectory(
        prompt="Explain AI safety.",
        steps=[step],
        final_response="AI safety is important.",
    )

    data = trajectory.to_dict()

    assert data["prompt"] == "Explain AI safety."

    assert len(data["steps"]) == 1

    assert data["final_response"] == (
        "AI safety is important."
    )


# ============================================================
# 3. StructuredIntrospector Tests
# ============================================================

def test_structured_introspector_generates_candidates():

    def candidate_function(
        prompt,
        previous_steps,
        num_candidates,
    ):

        return [
            ReasoningStep(
                stage="problem_framing",
                title=f"Candidate {index}",
                explanation=f"Analysis {index}",
            )
            for index in range(num_candidates)
        ]

    introspector = StructuredIntrospector(
        candidate_function=candidate_function
    )

    candidates = introspector.generate_candidates(
        prompt="Explain AI safety.",
        previous_steps=(),
        num_candidates=3,
    )

    assert len(candidates) == 3

    assert all(
        isinstance(step, ReasoningStep)
        for step in candidates
    )


def test_introspector_receives_previous_steps():

    previous_step = ReasoningStep(
        stage="problem_framing",
        title="Task",
        explanation="Identify the task.",
    )

    def candidate_function(
        prompt,
        previous_steps,
        num_candidates,
    ):

        assert prompt == "Explain AI safety."

        assert previous_steps[0] == previous_step

        return [
            ReasoningStep(
                stage="risk_policy_analysis",
                title="Risk analysis",
                explanation="Evaluate risks.",
            )
        ]

    introspector = StructuredIntrospector(
        candidate_function=candidate_function
    )

    candidates = introspector.generate_candidates(
        prompt="Explain AI safety.",
        previous_steps=[previous_step],
        num_candidates=1,
    )

    assert len(candidates) == 1


def test_introspector_rejects_invalid_candidate_function():

    introspector = StructuredIntrospector(
        candidate_function=lambda *args: []
    )

    with pytest.raises(ValueError):

        introspector.generate_candidates(
            prompt="",
            previous_steps=(),
            num_candidates=1,
        )
