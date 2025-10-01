"""
Evaluation metrics for trajectory criticism.

Provides metrics to evaluate critic performance and trajectory quality
in both open-loop and closed-loop settings.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import spearmanr, pearsonr
import warnings

from .physics_checks import PhysicsChecker, PhysicsConfig


@dataclass
class CriticMetrics:
    """Container for critic evaluation metrics."""
    
    # Classification metrics (risk prediction)
    risk_auroc: float = 0.0
    risk_auprc: float = 0.0
    risk_accuracy: float = 0.0
    
    # Regression metrics (score prediction)
    score_correlation: float = 0.0
    score_spearman: float = 0.0
    score_mae: float = 0.0
    score_rmse: float = 0.0
    
    # Ranking metrics
    ranking_accuracy: float = 0.0
    top_k_accuracy: float = 0.0
    
    # Physics consistency
    physics_correlation: float = 0.0
    
    # Closed-loop metrics (if available)
    collision_rate: Optional[float] = None
    off_route_rate: Optional[float] = None
    avg_jerk: Optional[float] = None
    avg_progress: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                result[field_name] = float(field_value)
        return result
        
    def __str__(self) -> str:
        """String representation of metrics."""
        lines = ["Critic Evaluation Metrics:"]
        lines.append(f"  Risk AUROC: {self.risk_auroc:.3f}")
        lines.append(f"  Score Correlation: {self.score_correlation:.3f}")
        lines.append(f"  Ranking Accuracy: {self.ranking_accuracy:.3f}")
        lines.append(f"  Physics Correlation: {self.physics_correlation:.3f}")
        
        if self.collision_rate is not None:
            lines.append(f"  Collision Rate: {self.collision_rate:.3f}")
        if self.off_route_rate is not None:
            lines.append(f"  Off-Route Rate: {self.off_route_rate:.3f}")
            
        return "\n".join(lines)


class CriticEvaluator:
    """
    Evaluator for trajectory critic models.
    
    Computes various metrics to assess critic performance including
    classification accuracy, regression quality, and ranking performance.
    """
    
    def __init__(self, physics_config: PhysicsConfig = None):
        """
        Initialize evaluator.
        
        Args:
            physics_config: Configuration for physics-based evaluation
        """
        self.physics_checker = PhysicsChecker(physics_config or PhysicsConfig())
        
    def evaluate_critic(
        self,
        predictions: List[Dict[str, float]],
        ground_truth: List[Dict[str, float]],
        trajectory_groups: Optional[List[List[int]]] = None
    ) -> CriticMetrics:
        """
        Evaluate critic predictions against ground truth.
        
        Args:
            predictions: List of predicted scores for each trajectory
            ground_truth: List of ground truth scores for each trajectory
            trajectory_groups: Optional grouping of trajectories for ranking evaluation
            
        Returns:
            Evaluation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
            
        metrics = CriticMetrics()
        
        # Extract score arrays
        pred_risk = np.array([p.get("risk", 0.0) for p in predictions])
        pred_comfort = np.array([p.get("comfort", 0.0) for p in predictions])
        pred_progress = np.array([p.get("progress", 0.0) for p in predictions])
        pred_composite = np.array([p.get("composite", 0.0) for p in predictions])
        
        gt_risk = np.array([g.get("risk", 0.0) for g in ground_truth])
        gt_comfort = np.array([g.get("comfort", 0.0) for g in ground_truth])
        gt_progress = np.array([g.get("progress", 0.0) for g in ground_truth])
        gt_composite = np.array([g.get("composite", 0.0) for g in ground_truth])
        
        # Classification metrics (risk as binary classification)
        metrics.risk_auroc = self._compute_auroc(pred_risk, gt_risk)
        metrics.risk_auprc = self._compute_auprc(pred_risk, gt_risk)
        metrics.risk_accuracy = self._compute_accuracy(pred_risk, gt_risk)
        
        # Regression metrics (composite score)
        metrics.score_correlation = self._compute_correlation(pred_composite, gt_composite)
        metrics.score_spearman = self._compute_spearman(pred_composite, gt_composite)
        metrics.score_mae = self._compute_mae(pred_composite, gt_composite)
        metrics.score_rmse = self._compute_rmse(pred_composite, gt_composite)
        
        # Ranking metrics
        if trajectory_groups:
            metrics.ranking_accuracy = self._compute_ranking_accuracy(
                pred_composite, gt_composite, trajectory_groups
            )
            metrics.top_k_accuracy = self._compute_top_k_accuracy(
                pred_composite, gt_composite, trajectory_groups, k=3
            )
        else:
            # Global ranking if no groups provided
            metrics.ranking_accuracy = self._compute_global_ranking_accuracy(
                pred_composite, gt_composite
            )
            
        # Physics consistency
        metrics.physics_correlation = self._compute_physics_consistency(
            predictions, ground_truth
        )
        
        return metrics
        
    def evaluate_physics_labels(
        self,
        trajectories: List,
        scenes: List,
        human_labels: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate quality of physics-based pseudo-labels.
        
        Args:
            trajectories: List of trajectory candidates
            scenes: List of scene data
            human_labels: Optional human annotations for comparison
            
        Returns:
            Dictionary of physics label quality metrics
        """
        # Generate physics labels
        physics_labels = []
        for traj, scene in zip(trajectories, scenes):
            label = self.physics_checker.evaluate_trajectory(traj, scene)
            physics_labels.append(label)
            
        metrics = {}
        
        # Internal consistency metrics
        risk_scores = [l["risk"] for l in physics_labels]
        comfort_scores = [l["comfort"] for l in physics_labels]
        progress_scores = [l["progress"] for l in physics_labels]
        
        metrics["risk_variance"] = float(np.var(risk_scores))
        metrics["comfort_variance"] = float(np.var(comfort_scores))
        metrics["progress_variance"] = float(np.var(progress_scores))
        
        # Score distribution metrics
        metrics["risk_mean"] = float(np.mean(risk_scores))
        metrics["comfort_mean"] = float(np.mean(comfort_scores))
        metrics["progress_mean"] = float(np.mean(progress_scores))
        
        # Compare with human labels if available
        if human_labels:
            human_risk = [l.get("risk", 0.0) for l in human_labels]
            human_comfort = [l.get("comfort", 0.0) for l in human_labels]
            human_progress = [l.get("progress", 0.0) for l in human_labels]
            
            metrics["human_risk_correlation"] = self._compute_correlation(
                np.array(risk_scores), np.array(human_risk)
            )
            metrics["human_comfort_correlation"] = self._compute_correlation(
                np.array(comfort_scores), np.array(human_comfort)
            )
            metrics["human_progress_correlation"] = self._compute_correlation(
                np.array(progress_scores), np.array(human_progress)
            )
            
        return metrics
        
    def _compute_auroc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        try:
            # Convert to binary classification (threshold at 0.5)
            binary_targets = (targets > 0.5).astype(int)
            
            if len(np.unique(binary_targets)) < 2:
                return 0.5  # No positive or negative examples
                
            return float(roc_auc_score(binary_targets, predictions))
        except Exception:
            return 0.0
            
    def _compute_auprc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve."""
        try:
            binary_targets = (targets > 0.5).astype(int)
            
            if len(np.unique(binary_targets)) < 2:
                return 0.5
                
            precision, recall, _ = precision_recall_curve(binary_targets, predictions)
            return float(auc(recall, precision))
        except Exception:
            return 0.0
            
    def _compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute binary classification accuracy."""
        try:
            pred_binary = (predictions > 0.5).astype(int)
            target_binary = (targets > 0.5).astype(int)
            return float(np.mean(pred_binary == target_binary))
        except Exception:
            return 0.0
            
    def _compute_correlation(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        try:
            if len(predictions) < 2:
                return 0.0
            corr, _ = pearsonr(predictions, targets)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
            
    def _compute_spearman(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Spearman rank correlation."""
        try:
            if len(predictions) < 2:
                return 0.0
            corr, _ = spearmanr(predictions, targets)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
            
    def _compute_mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(np.mean(np.abs(predictions - targets)))
        
    def _compute_rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Root Mean Square Error."""
        return float(np.sqrt(np.mean((predictions - targets) ** 2)))
        
    def _compute_ranking_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        groups: List[List[int]]
    ) -> float:
        """Compute ranking accuracy within groups."""
        correct_rankings = 0
        total_rankings = 0
        
        for group in groups:
            if len(group) < 2:
                continue
                
            group_preds = predictions[group]
            group_targets = targets[group]
            
            # Get ranking orders
            pred_order = np.argsort(-group_preds)  # Descending order
            target_order = np.argsort(-group_targets)
            
            # Check if best predicted matches best target
            if pred_order[0] == target_order[0]:
                correct_rankings += 1
            total_rankings += 1
            
        return correct_rankings / total_rankings if total_rankings > 0 else 0.0
        
    def _compute_top_k_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        groups: List[List[int]],
        k: int = 3
    ) -> float:
        """Compute top-k ranking accuracy within groups."""
        correct_rankings = 0
        total_rankings = 0
        
        for group in groups:
            if len(group) < k:
                continue
                
            group_preds = predictions[group]
            group_targets = targets[group]
            
            # Get top-k indices
            pred_top_k = np.argsort(-group_preds)[:k]
            target_top_k = np.argsort(-group_targets)[:k]
            
            # Check overlap
            overlap = len(set(pred_top_k) & set(target_top_k))
            correct_rankings += overlap / k
            total_rankings += 1
            
        return correct_rankings / total_rankings if total_rankings > 0 else 0.0
        
    def _compute_global_ranking_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute global ranking correlation."""
        try:
            return self._compute_spearman(predictions, targets)
        except Exception:
            return 0.0
            
    def _compute_physics_consistency(
        self,
        predictions: List[Dict[str, float]],
        ground_truth: List[Dict[str, float]]
    ) -> float:
        """Compute consistency with physics-based expectations."""
        try:
            # Extract individual metric correlations
            correlations = []
            
            for metric in ["risk", "comfort", "progress"]:
                pred_values = np.array([p.get(metric, 0.0) for p in predictions])
                gt_values = np.array([g.get(metric, 0.0) for g in ground_truth])
                
                corr = self._compute_correlation(pred_values, gt_values)
                correlations.append(corr)
                
            return float(np.mean(correlations))
        except Exception:
            return 0.0


def compute_closed_loop_metrics(
    rollout_results: List[Dict],
    physics_config: PhysicsConfig = None
) -> Dict[str, float]:
    """
    Compute closed-loop evaluation metrics from rollout results.
    
    Args:
        rollout_results: List of rollout result dictionaries
        physics_config: Configuration for physics evaluation
        
    Returns:
        Dictionary of closed-loop metrics
    """
    if not rollout_results:
        return {}
        
    config = physics_config or PhysicsConfig()
    
    # Extract metrics from rollouts
    collision_flags = []
    off_route_flags = []
    jerk_values = []
    progress_values = []
    
    for result in rollout_results:
        # Collision detection
        collisions = result.get("collisions", [])
        collision_flags.append(any(collisions))
        
        # Off-route detection
        off_route = result.get("off_route", [])
        off_route_flags.append(any(off_route))
        
        # Jerk computation
        trajectory = result.get("trajectory", [])
        if len(trajectory) >= 3:
            trajectory_array = np.array(trajectory)
            jerks = compute_jerk(trajectory_array)
            if len(jerks) > 0:
                jerk_values.append(np.mean(jerks))
                
        # Progress computation
        progress = result.get("progress_score", 0.0)
        progress_values.append(progress)
        
    # Compute aggregate metrics
    metrics = {}
    
    if collision_flags:
        metrics["collision_rate"] = float(np.mean(collision_flags))
        
    if off_route_flags:
        metrics["off_route_rate"] = float(np.mean(off_route_flags))
        
    if jerk_values:
        metrics["avg_jerk"] = float(np.mean(jerk_values))
        metrics["max_jerk"] = float(np.max(jerk_values))
        
    if progress_values:
        metrics["avg_progress"] = float(np.mean(progress_values))
        metrics["min_progress"] = float(np.min(progress_values))
        
    return metrics


def compute_jerk(trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Compute jerk magnitudes from trajectory.
    
    Args:
        trajectory: Trajectory array [T, 4] with [x, y, vx, vy]
        dt: Time step in seconds
        
    Returns:
        Array of jerk magnitudes
    """
    if len(trajectory) < 3:
        return np.array([])
        
    # Extract velocities
    velocities = trajectory[:, 2:4]
    
    # Compute accelerations
    accelerations = np.diff(velocities, axis=0) / dt
    
    if len(accelerations) < 2:
        return np.array([])
        
    # Compute jerks
    jerks = np.diff(accelerations, axis=0) / dt
    
    # Return magnitudes
    return np.linalg.norm(jerks, axis=1)