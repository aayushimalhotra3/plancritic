"""
PlanCritic: A learned trajectory evaluator for autonomous vehicles.

Tagline: "Score twice, drive once."
"""

__version__ = "0.1.0"
__author__ = "PlanCritic Team"

from .models.critic import TrajectoryCritic
from .models.encoders import LaneGraphEncoder, StateEncoder, TrajectoryEncoder

__all__ = [
    "TrajectoryCritic",
    "LaneGraphEncoder", 
    "StateEncoder",
    "TrajectoryEncoder",
]