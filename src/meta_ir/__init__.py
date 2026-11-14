
"""
META-IR
Meta-Introspective Reasoning for Adaptive Safety Alignment in LLM.

Research implementation for IEEE ICME 2026.
"""

__version__ = "0.1.0"
__author__ = "Mahabubur Rahman Miraj"

from meta_ir.reasoning.introspection import (
    ReasoningStep,
    IntrospectionTrajectory,
    StructuredIntrospector,
)

__all__ = [
    "__version__",
    "ReasoningStep",
    "IntrospectionTrajectory",
    "StructuredIntrospector",
]
