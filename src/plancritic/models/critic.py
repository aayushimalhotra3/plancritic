"""
TrajectoryCritic: Neural network model for scoring AV trajectory candidates.

The critic evaluates trajectories on three dimensions:
- Risk: Collision probability and safety violations
- Comfort: Jerk and smoothness metrics  
- Progress: Route advancement and efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TrajectoryCritic(nn.Module):
    """
    Multi-head neural network that scores trajectory candidates.
    
    Takes ego state, lane graph features, and candidate trajectory features
    as input and outputs risk, comfort, progress, and composite scores.
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        lane_dim: int = 64, 
        cand_dim: int = 64,
        hidden: int = 128,
        dropout: float = 0.1,
        composite_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the TrajectoryCritic.
        
        Args:
            state_dim: Dimension of ego state features
            lane_dim: Dimension of lane graph features
            cand_dim: Dimension of candidate trajectory features
            hidden: Hidden layer dimension
            dropout: Dropout probability
            composite_weights: Weights for composite score calculation
        """
        super().__init__()
        
        # Feature encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lane_encoder = nn.Sequential(
            nn.Linear(lane_dim, hidden),
            nn.ReLU(), 
            nn.Dropout(dropout)
        )
        
        self.cand_encoder = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Scoring heads
        self.heads = nn.ModuleDict({
            "risk": nn.Linear(hidden, 1),
            "comfort": nn.Linear(hidden, 1), 
            "progress": nn.Linear(hidden, 1)
        })
        
        # Composite score weights
        if composite_weights is None:
            composite_weights = {"risk": 0.5, "comfort": 0.2, "progress": 0.3}
        self.register_buffer("composite_weights", torch.tensor([
            composite_weights["risk"],
            composite_weights["comfort"], 
            composite_weights["progress"]
        ]))
        
    def forward(
        self, 
        state_feats: torch.Tensor,
        lane_feats: torch.Tensor, 
        cand_feats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the critic network.
        
        Args:
            state_feats: Ego state features [B, state_dim]
            lane_feats: Lane graph features [B, lane_dim] 
            cand_feats: Candidate trajectory features [B, cand_dim]
            
        Returns:
            Dictionary with risk, comfort, progress, and composite scores
        """
        # Encode features
        state_h = self.state_encoder(state_feats)
        lane_h = self.lane_encoder(lane_feats)
        cand_h = self.cand_encoder(cand_feats)
        
        # Fuse features
        fused = torch.cat([state_h, lane_h, cand_h], dim=-1)
        z = self.fusion(fused)
        
        # Compute individual scores
        risk = torch.sigmoid(self.heads["risk"](z))
        comfort = torch.sigmoid(self.heads["comfort"](z))
        progress = torch.sigmoid(self.heads["progress"](z))
        
        # Compute composite score
        # Lower risk & discomfort, higher progress is better
        scores = torch.cat([1 - risk, 1 - comfort, progress], dim=-1)
        composite = torch.sum(scores * self.composite_weights, dim=-1, keepdim=True)
        
        return {
            "risk": risk,
            "comfort": comfort, 
            "progress": progress,
            "score": composite
        }
    
    def set_composite_weights(self, weights: Dict[str, float]) -> None:
        """Update composite score weights."""
        self.composite_weights = torch.tensor([
            weights["risk"],
            weights["comfort"],
            weights["progress"] 
        ], device=self.composite_weights.device)


class MultiCandidateCritic(nn.Module):
    """
    Extension of TrajectoryCritic that handles multiple candidates efficiently.
    
    Processes N candidate trajectories in parallel and returns scores for each.
    """
    
    def __init__(self, critic: TrajectoryCritic):
        """Initialize with a base critic model."""
        super().__init__()
        self.critic = critic
        
    def forward(
        self,
        state_feats: torch.Tensor,
        lane_feats: torch.Tensor,
        cand_feats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-candidate critic.
        
        Args:
            state_feats: Ego state features [B, state_dim]
            lane_feats: Lane graph features [B, lane_dim]
            cand_feats: Candidate trajectory features [B, N, cand_dim] or [B, cand_dim]
            
        Returns:
            Dictionary with scores for each candidate [B, N, 1]
        """
        # Handle both 2D and 3D candidate features
        if len(cand_feats.shape) == 2:
            # If cand_feats is [B, cand_dim], add a candidate dimension
            B, cand_dim = cand_feats.shape
            N = 1  # Single candidate
            cand_feats = cand_feats.unsqueeze(1)  # [B, 1, cand_dim]
        else:
            B, N, cand_dim = cand_feats.shape
        
        # Expand state and lane features for each candidate
        state_expanded = state_feats.unsqueeze(1).expand(B, N, -1)
        lane_expanded = lane_feats.unsqueeze(1).expand(B, N, -1)
        
        # Reshape for batch processing
        state_flat = state_expanded.reshape(B * N, -1)
        lane_flat = lane_expanded.reshape(B * N, -1)
        cand_flat = cand_feats.reshape(B * N, -1)
        
        # Score all candidates
        scores = self.critic(state_flat, lane_flat, cand_flat)
        
        # Reshape back to [B, N, 1]
        result = {}
        for key, value in scores.items():
            result[key] = value.reshape(B, N, 1)
            
        return result