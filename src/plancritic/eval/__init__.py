"""Evaluation utilities for trajectory criticism."""

from .physics_checks import (
    PhysicsChecker, 
    PhysicsConfig, 
    compute_ttc, 
    compute_jerk, 
    check_map_collision
)
from .metrics import (
    CriticMetrics, 
    CriticEvaluator, 
    compute_closed_loop_metrics
)

__all__ = [
    "PhysicsChecker",
    "PhysicsConfig", 
    "compute_ttc",
    "compute_jerk",
    "check_map_collision",
    "CriticMetrics",
    "CriticEvaluator",
    "compute_closed_loop_metrics"
]