"""
Loss functions for training trajectory critics.

Includes physics-based losses for self-supervision and
standard losses for supervised learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class CriticLoss(nn.Module):
    """
    Multi-task loss for trajectory criticism.
    
    Combines losses for risk, comfort, and progress prediction
    with optional physics-based regularization.
    """
    
    def __init__(
        self,
        risk_weight: float = 1.0,
        comfort_weight: float = 1.0,
        progress_weight: float = 1.0,
        calibration_weight: float = 0.1,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Initialize multi-task critic loss.
        
        Args:
            risk_weight: Weight for risk prediction loss
            comfort_weight: Weight for comfort prediction loss  
            progress_weight: Weight for progress prediction loss
            calibration_weight: Weight for score calibration loss
            use_focal_loss: Whether to use focal loss for imbalanced data
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        
        self.risk_weight = risk_weight
        self.comfort_weight = comfort_weight
        self.progress_weight = progress_weight
        self.calibration_weight = calibration_weight
        
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def focal_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        alpha: float = 0.25, 
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute focal loss for addressing class imbalance.
        
        Args:
            pred: Predicted probabilities [B, 1]
            target: Target labels [B, 1]
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = alpha * (1 - pt) ** gamma
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()
        
    def calibration_loss(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encourage well-calibrated probability predictions.
        
        Args:
            scores: Dictionary of predicted scores
            
        Returns:
            Calibration loss encouraging scores to be well-distributed
        """
        calibration_losses = []
        
        for score_name, score_tensor in scores.items():
            if score_name == "score":  # Skip composite score
                continue
                
            # Encourage scores to use full [0, 1] range
            mean_score = score_tensor.mean()
            var_score = score_tensor.var()
            
            # Penalize scores that are too concentrated
            range_loss = F.mse_loss(mean_score, torch.tensor(0.5, device=mean_score.device))
            var_loss = F.mse_loss(var_score, torch.tensor(0.1, device=var_score.device))
            
            calibration_losses.append(range_loss + var_loss)
            
        return torch.stack(calibration_losses).mean() if calibration_losses else torch.tensor(0.0)
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task critic loss.
        
        Args:
            predictions: Dictionary of predicted scores
            targets: Dictionary of target scores
            weights: Optional sample weights [B, 1]
            
        Returns:
            Total loss and dictionary of individual losses
        """
        losses = {}
        
        # Risk prediction loss
        if "risk" in predictions and "risk" in targets:
            if self.use_focal_loss:
                risk_loss = self.focal_loss(
                    predictions["risk"], targets["risk"],
                    self.focal_alpha, self.focal_gamma
                )
            else:
                risk_loss = F.binary_cross_entropy(
                    predictions["risk"], targets["risk"], reduction='none'
                )
                if weights is not None:
                    risk_loss = (risk_loss * weights).mean()
                else:
                    risk_loss = risk_loss.mean()
            losses["risk"] = risk_loss * self.risk_weight
            
        # Comfort prediction loss
        if "comfort" in predictions and "comfort" in targets:
            comfort_loss = F.mse_loss(predictions["comfort"], targets["comfort"], reduction='none')
            if weights is not None:
                comfort_loss = (comfort_loss * weights).mean()
            else:
                comfort_loss = comfort_loss.mean()
            losses["comfort"] = comfort_loss * self.comfort_weight
            
        # Progress prediction loss  
        if "progress" in predictions and "progress" in targets:
            progress_loss = F.mse_loss(predictions["progress"], targets["progress"], reduction='none')
            if weights is not None:
                progress_loss = (progress_loss * weights).mean()
            else:
                progress_loss = progress_loss.mean()
            losses["progress"] = progress_loss * self.progress_weight
            
        # Calibration loss
        if self.calibration_weight > 0:
            calib_loss = self.calibration_loss(predictions)
            losses["calibration"] = calib_loss * self.calibration_weight
            
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses


class PhysicsLoss(nn.Module):
    """
    Physics-based loss for self-supervised learning.
    
    Generates pseudo-labels from physics checks and trains
    the critic to match these physics-derived scores.
    """
    
    def __init__(
        self,
        ttc_threshold: float = 3.0,
        jerk_threshold: float = 2.0,
        collision_weight: float = 10.0,
        comfort_weight: float = 1.0,
        progress_weight: float = 1.0
    ):
        """
        Initialize physics-based loss.
        
        Args:
            ttc_threshold: Time-to-collision threshold (seconds)
            jerk_threshold: Jerk threshold (m/s³)
            collision_weight: Weight for collision penalties
            comfort_weight: Weight for comfort metrics
            progress_weight: Weight for progress metrics
        """
        super().__init__()
        
        self.ttc_threshold = ttc_threshold
        self.jerk_threshold = jerk_threshold
        self.collision_weight = collision_weight
        self.comfort_weight = comfort_weight
        self.progress_weight = progress_weight
        
    def compute_ttc_risk(
        self, 
        ego_trajectory: torch.Tensor,
        agent_states: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute collision risk based on time-to-collision.
        
        Args:
            ego_trajectory: Ego trajectory waypoints [B, T, 4] (x, y, vx, vy)
            agent_states: Agent states [B, N, 4] (x, y, vx, vy)
            agent_mask: Valid agent mask [B, N]
            
        Returns:
            Risk scores based on minimum TTC [B, 1]
        """
        B, T, _ = ego_trajectory.shape
        B, N, _ = agent_states.shape
        
        # Expand dimensions for pairwise computation
        ego_pos = ego_trajectory[:, :, :2].unsqueeze(2)  # [B, T, 1, 2]
        ego_vel = ego_trajectory[:, :, 2:].unsqueeze(2)  # [B, T, 1, 2]
        
        agent_pos = agent_states[:, :, :2].unsqueeze(1)  # [B, 1, N, 2]
        agent_vel = agent_states[:, :, 2:].unsqueeze(1)  # [B, 1, N, 2]
        
        # Relative position and velocity
        rel_pos = ego_pos - agent_pos  # [B, T, N, 2]
        rel_vel = ego_vel - agent_vel  # [B, T, N, 2]
        
        # Time to collision calculation
        # TTC = -dot(rel_pos, rel_vel) / ||rel_vel||²
        rel_pos_dot_vel = torch.sum(rel_pos * rel_vel, dim=-1)  # [B, T, N]
        rel_vel_norm_sq = torch.sum(rel_vel ** 2, dim=-1)  # [B, T, N]
        
        # Avoid division by zero
        ttc = -rel_pos_dot_vel / (rel_vel_norm_sq + 1e-8)
        
        # Only consider positive TTC (approaching)
        ttc = torch.where(ttc > 0, ttc, float('inf'))
        
        # Apply agent mask
        if agent_mask is not None:
            ttc = ttc.masked_fill(~agent_mask.unsqueeze(1), float('inf'))
            
        # Minimum TTC across all agents and time steps
        min_ttc = torch.min(ttc.view(B, -1), dim=1)[0]  # [B]
        
        # Convert to risk score (sigmoid of inverse TTC)
        risk = torch.sigmoid(self.ttc_threshold / (min_ttc + 1e-8))
        
        return risk.unsqueeze(-1)
        
    def compute_jerk_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute comfort penalty based on jerk (third derivative of position).
        
        Args:
            trajectory: Trajectory waypoints [B, T, 4] (x, y, vx, vy)
            
        Returns:
            Comfort penalty scores [B, 1]
        """
        # Extract velocities
        velocities = trajectory[:, :, 2:]  # [B, T, 2]
        
        # Compute accelerations (first derivative of velocity)
        dt = 0.1  # Assume 10Hz sampling
        accelerations = torch.diff(velocities, dim=1) / dt  # [B, T-1, 2]
        
        # Compute jerk (second derivative of velocity)
        jerk = torch.diff(accelerations, dim=1) / dt  # [B, T-2, 2]
        
        # Jerk magnitude
        jerk_magnitude = torch.norm(jerk, dim=-1)  # [B, T-2]
        
        # Average jerk over trajectory
        avg_jerk = torch.mean(jerk_magnitude, dim=1)  # [B]
        
        # Convert to comfort penalty (sigmoid of normalized jerk)
        comfort_penalty = torch.sigmoid(avg_jerk / self.jerk_threshold)
        
        return comfort_penalty.unsqueeze(-1)
        
    def compute_progress_score(
        self, 
        trajectory: torch.Tensor,
        route_waypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute progress score based on route advancement.
        
        Args:
            trajectory: Ego trajectory [B, T, 4]
            route_waypoints: Reference route [B, R, 2]
            
        Returns:
            Progress scores [B, 1]
        """
        # Final position of trajectory
        final_pos = trajectory[:, -1, :2]  # [B, 2]
        
        # Distance to each route waypoint
        distances = torch.norm(
            final_pos.unsqueeze(1) - route_waypoints, dim=-1
        )  # [B, R]
        
        # Find closest waypoint index
        closest_idx = torch.argmin(distances, dim=1)  # [B]
        
        # Progress as normalized waypoint index
        progress = closest_idx.float() / (route_waypoints.shape[1] - 1)
        
        return progress.unsqueeze(-1)
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        ego_trajectory: torch.Tensor,
        agent_states: torch.Tensor,
        route_waypoints: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics-based loss.
        
        Args:
            predictions: Model predictions
            ego_trajectory: Ego trajectory waypoints
            agent_states: Nearby agent states
            route_waypoints: Reference route
            agent_mask: Valid agent mask
            
        Returns:
            Total loss and individual loss components
        """
        losses = {}
        
        # Compute physics-based pseudo-labels
        risk_target = self.compute_ttc_risk(ego_trajectory, agent_states, agent_mask)
        comfort_target = self.compute_jerk_penalty(ego_trajectory)
        progress_target = self.compute_progress_score(ego_trajectory, route_waypoints)
        
        # Risk loss (collision avoidance)
        if "risk" in predictions:
            risk_loss = F.binary_cross_entropy(predictions["risk"], risk_target)
            losses["physics_risk"] = risk_loss * self.collision_weight
            
        # Comfort loss (jerk penalty)
        if "comfort" in predictions:
            comfort_loss = F.mse_loss(predictions["comfort"], comfort_target)
            losses["physics_comfort"] = comfort_loss * self.comfort_weight
            
        # Progress loss (route advancement)
        if "progress" in predictions:
            progress_loss = F.mse_loss(predictions["progress"], progress_target)
            losses["physics_progress"] = progress_loss * self.progress_weight
            
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning trajectory preferences.
    
    Encourages the model to score better trajectories higher
    than worse trajectories in a pairwise manner.
    """
    
    def __init__(self, margin: float = 0.1, temperature: float = 0.1):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for pairwise ranking loss
            temperature: Temperature for softmax normalization
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(
        self,
        scores: torch.Tensor,
        preferences: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss from trajectory preferences.
        
        Args:
            scores: Predicted scores [B, N] for N candidates
            preferences: Preference rankings [B, N] (higher = better)
            
        Returns:
            Contrastive loss encouraging preference ordering
        """
        B, N = scores.shape
        
        # Pairwise score differences
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B, N, N]
        
        # Pairwise preference differences  
        pref_diff = preferences.unsqueeze(2) - preferences.unsqueeze(1)  # [B, N, N]
        
        # Create preference targets (1 if i > j, -1 if i < j, 0 if equal)
        targets = torch.sign(pref_diff)
        
        # Ranking loss: encourage score_diff to match preference order
        loss = F.margin_ranking_loss(
            score_diff.view(-1),
            torch.zeros_like(score_diff.view(-1)),
            targets.view(-1),
            margin=self.margin,
            reduction='mean'
        )
        
        return loss