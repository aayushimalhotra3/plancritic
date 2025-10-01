"""Command-line interface tools for PlanCritic.

This module provides CLI tools for training, evaluation, and data processing.
"""

from .train import Trainer, TrainingConfig
from .score import TrajectoryScorer
from .export import TrajectoryExporter

__all__ = [
    "Trainer",
    "TrainingConfig", 
    "TrajectoryScorer",
    "TrajectoryExporter"
]