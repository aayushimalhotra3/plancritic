"""
Feature encoders for different input modalities in trajectory criticism.

Includes encoders for:
- Ego vehicle state (position, velocity, heading, etc.)
- Lane graph structure (polylines, connectivity, attributes)
- Trajectory candidates (waypoints, velocities, curvature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

from torch_geometric.nn import GCNConv, global_mean_pool


class StateEncoder(nn.Module):
    """
    Encodes ego vehicle state into a fixed-size representation.
    
    Handles position, velocity, acceleration, heading, and other
    vehicle dynamics features.
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # [x, y, vx, vy, ax, ay, heading, yaw_rate]
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode ego state.
        
        Args:
            state: Ego state tensor [B, input_dim]
            
        Returns:
            Encoded state features [B, output_dim]
        """
        return self.encoder(state)


class TrajectoryEncoder(nn.Module):
    """
    Encodes trajectory candidates into fixed-size representations.
    
    Uses 1D convolutions over the temporal dimension to capture
    trajectory patterns and dynamics.
    """
    
    def __init__(
        self,
        waypoint_dim: int = 4,  # [x, y, vx, vy] per waypoint
        seq_len: int = 80,      # Number of waypoints
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.waypoint_dim = waypoint_dim
        self.seq_len = seq_len
        
        # 1D convolutions over time
        self.conv1 = nn.Conv1d(waypoint_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        
        # Global pooling and final projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory candidate.
        
        Args:
            trajectory: Trajectory waypoints [B, seq_len, waypoint_dim]
            
        Returns:
            Encoded trajectory features [B, output_dim]
        """
        # Transpose for conv1d: [B, waypoint_dim, seq_len]
        x = trajectory.transpose(1, 2)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling and projection
        x = self.pool(x).squeeze(-1)  # [B, hidden_dim]
        x = self.dropout(x)
        x = self.projection(x)
        
        return x


class LaneGraphEncoder(nn.Module):
    """
    Encodes lane graph structure using GCN convolutions from torch-geometric.

    Accepts node features + sparse edge_index, applies GCNConv layers with
    residual connections, pools per-graph, and projects to output_dim.
    """

    def __init__(
        self,
        node_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: Node features [N, node_dim]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment vector [N] (from PyG batching)

        Returns:
            Graph-level features [B, output_dim]
        """
        x = F.relu(self.node_proj(node_features))

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x) + residual

        pooled = global_mean_pool(x, batch)
        return self.output_proj(pooled)


class PolylineEncoder(nn.Module):
    """
    Encodes individual polylines (lane centerlines, boundaries) into vectors.
    
    Uses 1D convolutions over the polyline points to capture
    geometric patterns and curvature information.
    """
    
    def __init__(
        self,
        point_dim: int = 3,     # [x, y, heading] per point
        max_points: int = 20,   # Maximum points per polyline
        hidden_dim: int = 32,
        output_dim: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.point_dim = point_dim
        self.max_points = max_points
        
        # 1D convolutions over polyline points
        self.conv1 = nn.Conv1d(point_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, polyline: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode polyline.
        
        Args:
            polyline: Polyline points [B, max_points, point_dim]
            mask: Valid point mask [B, max_points] (optional)
            
        Returns:
            Encoded polyline features [B, output_dim]
        """
        # Transpose for conv1d: [B, point_dim, max_points]
        x = polyline.transpose(1, 2)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Transpose back: [B, max_points, hidden_dim]
        x = x.transpose(1, 2)
        
        # Attention pooling
        attn_weights = self.attention(x)  # [B, max_points, 1]
        
        if mask is not None:
            # Mask invalid points
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            
        attn_weights = F.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)  # [B, hidden_dim]
        
        # Final projection
        x = self.dropout(x)
        x = self.projection(x)
        
        return x


class AgentEncoder(nn.Module):
    """
    Encodes nearby agent states and trajectories.
    
    Processes multiple agents in the scene and creates
    a scene-level representation of agent interactions.
    """
    
    def __init__(
        self,
        agent_dim: int = 8,     # [x, y, vx, vy, ax, ay, heading, yaw_rate]
        max_agents: int = 32,   # Maximum agents to consider
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_agents = max_agents
        
        # Agent feature encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Self-attention over agents
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self, 
        agents: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode agent states.
        
        Args:
            agents: Agent states [B, max_agents, agent_dim]
            mask: Valid agent mask [B, max_agents] (optional)
            
        Returns:
            Scene-level agent encoding [B, output_dim]
        """
        # Encode individual agents
        x = self.agent_encoder(agents)  # [B, max_agents, hidden_dim]
        
        # Self-attention over agents
        if mask is not None:
            # Convert mask for attention (True = valid, False = invalid)
            attn_mask = ~mask  # Invert for attention mask
        else:
            attn_mask = None
            
        x, _ = self.self_attention(x, x, x, key_padding_mask=attn_mask)
        
        # Global pooling
        if mask is not None:
            # Masked mean pooling
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
            
        # Final projection
        x = self.output_proj(x)
        
        return x