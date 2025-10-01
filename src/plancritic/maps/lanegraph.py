"""
Lane graph representation and encoding for trajectory criticism.

Provides data structures and neural network encoders for processing
lane graph information from HD maps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class LaneType(Enum):
    """Lane type enumeration."""
    FREEWAY = "FREEWAY"
    SURFACE_STREET = "SURFACE_STREET"
    BIKE_LANE = "BIKE_LANE"
    BUS_LANE = "BUS_LANE"
    PARKING = "PARKING"
    SHOULDER = "SHOULDER"


class ConnectionType(Enum):
    """Lane connection type enumeration."""
    SUCCESSOR = "SUCCESSOR"
    PREDECESSOR = "PREDECESSOR"
    LEFT_NEIGHBOR = "LEFT_NEIGHBOR"
    RIGHT_NEIGHBOR = "RIGHT_NEIGHBOR"
    LANE_CHANGE = "LANE_CHANGE"


@dataclass
class LaneSegment:
    """Represents a single lane segment."""
    id: str
    centerline: np.ndarray  # [N, 2] waypoints
    left_boundary: np.ndarray  # [N, 2] left boundary points
    right_boundary: np.ndarray  # [N, 2] right boundary points
    speed_limit: float  # m/s
    lane_type: LaneType
    width: float  # meters
    
    def __post_init__(self):
        """Validate lane segment data."""
        if len(self.centerline) == 0:
            raise ValueError("Centerline cannot be empty")
        if self.speed_limit <= 0:
            raise ValueError("Speed limit must be positive")
        if self.width <= 0:
            raise ValueError("Lane width must be positive")


@dataclass
class LaneConnection:
    """Represents a connection between lane segments."""
    from_lane: str
    to_lane: str
    connection_type: ConnectionType
    cost: float = 1.0  # Cost for traversing this connection


class LaneGraph:
    """
    Lane graph representation for HD map data.
    
    Stores lane segments and their connections in a graph structure
    suitable for neural network processing.
    """
    
    def __init__(self):
        """Initialize empty lane graph."""
        self.lanes: Dict[str, LaneSegment] = {}
        self.connections: List[LaneConnection] = []
        self._adjacency_cache: Optional[Dict] = None
        
    def add_lane(self, lane: LaneSegment) -> None:
        """Add a lane segment to the graph."""
        self.lanes[lane.id] = lane
        self._adjacency_cache = None  # Invalidate cache
        
    def add_connection(self, connection: LaneConnection) -> None:
        """Add a connection between lanes."""
        # Validate that lanes exist
        if connection.from_lane not in self.lanes:
            raise ValueError(f"From lane {connection.from_lane} not found")
        if connection.to_lane not in self.lanes:
            raise ValueError(f"To lane {connection.to_lane} not found")
            
        self.connections.append(connection)
        self._adjacency_cache = None  # Invalidate cache
        
    def get_adjacent_lanes(self, lane_id: str) -> Dict[ConnectionType, List[str]]:
        """Get adjacent lanes by connection type."""
        if self._adjacency_cache is None:
            self._build_adjacency_cache()
            
        return self._adjacency_cache.get(lane_id, {})
        
    def get_lanes_in_radius(
        self, 
        center: np.ndarray, 
        radius: float
    ) -> List[str]:
        """Get lane IDs within radius of center point."""
        nearby_lanes = []
        
        for lane_id, lane in self.lanes.items():
            # Check if any point on centerline is within radius
            distances = np.linalg.norm(lane.centerline - center, axis=1)
            if np.min(distances) <= radius:
                nearby_lanes.append(lane_id)
                
        return nearby_lanes
        
    def to_tensor_representation(
        self, 
        max_lanes: int = 128,
        max_points_per_lane: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Convert lane graph to tensor representation for neural networks.
        
        Args:
            max_lanes: Maximum number of lanes to include
            max_points_per_lane: Maximum points per lane centerline
            
        Returns:
            Dictionary of tensors representing the lane graph
        """
        # Initialize tensors
        lane_features = torch.zeros(max_lanes, max_points_per_lane, 6)  # [x, y, heading, speed, width, type]
        lane_mask = torch.zeros(max_lanes, max_points_per_lane, dtype=torch.bool)
        lane_lengths = torch.zeros(max_lanes, dtype=torch.long)
        
        # Adjacency matrix for connections
        adjacency = torch.zeros(max_lanes, max_lanes)
        
        # Process lanes
        lane_ids = list(self.lanes.keys())[:max_lanes]
        lane_id_to_idx = {lane_id: i for i, lane_id in enumerate(lane_ids)}
        
        for i, lane_id in enumerate(lane_ids):
            lane = self.lanes[lane_id]
            
            # Resample centerline to fixed number of points
            centerline = self._resample_polyline(lane.centerline, max_points_per_lane)
            num_points = len(centerline)
            
            # Compute headings
            headings = self._compute_headings(centerline)
            
            # Fill features
            lane_features[i, :num_points, 0] = torch.from_numpy(centerline[:, 0])  # x
            lane_features[i, :num_points, 1] = torch.from_numpy(centerline[:, 1])  # y
            lane_features[i, :num_points, 2] = torch.from_numpy(headings)  # heading
            lane_features[i, :num_points, 3] = lane.speed_limit  # speed limit
            lane_features[i, :num_points, 4] = lane.width  # width
            lane_features[i, :num_points, 5] = self._lane_type_to_int(lane.lane_type)  # type
            
            # Set mask and length
            lane_mask[i, :num_points] = True
            lane_lengths[i] = num_points
            
        # Process connections
        for connection in self.connections:
            from_idx = lane_id_to_idx.get(connection.from_lane)
            to_idx = lane_id_to_idx.get(connection.to_lane)
            
            if from_idx is not None and to_idx is not None:
                adjacency[from_idx, to_idx] = 1.0 / connection.cost
                
        return {
            "lane_features": lane_features,
            "lane_mask": lane_mask,
            "lane_lengths": lane_lengths,
            "adjacency": adjacency,
            "lane_ids": lane_ids
        }
        
    def _build_adjacency_cache(self) -> None:
        """Build adjacency cache for fast lookups."""
        self._adjacency_cache = {}
        
        for lane_id in self.lanes:
            self._adjacency_cache[lane_id] = {conn_type: [] for conn_type in ConnectionType}
            
        for connection in self.connections:
            from_lane = connection.from_lane
            conn_type = connection.connection_type
            to_lane = connection.to_lane
            
            if from_lane in self._adjacency_cache:
                self._adjacency_cache[from_lane][conn_type].append(to_lane)
                
    def _resample_polyline(
        self, 
        polyline: np.ndarray, 
        num_points: int
    ) -> np.ndarray:
        """Resample polyline to fixed number of points."""
        if len(polyline) == 0:
            return np.zeros((0, 2))
            
        if len(polyline) == 1:
            return np.tile(polyline[0], (num_points, 1))
            
        # Compute cumulative distances
        distances = np.cumsum([0] + [np.linalg.norm(polyline[i+1] - polyline[i]) 
                                    for i in range(len(polyline)-1)])
        total_distance = distances[-1]
        
        if total_distance == 0:
            return np.tile(polyline[0], (num_points, 1))
            
        # Interpolate to get evenly spaced points
        target_distances = np.linspace(0, total_distance, num_points)
        
        resampled = np.zeros((num_points, 2))
        for i, target_dist in enumerate(target_distances):
            # Find segment containing target distance
            seg_idx = np.searchsorted(distances, target_dist) - 1
            seg_idx = max(0, min(seg_idx, len(polyline) - 2))
            
            # Interpolate within segment
            seg_start_dist = distances[seg_idx]
            seg_end_dist = distances[seg_idx + 1]
            
            if seg_end_dist > seg_start_dist:
                t = (target_dist - seg_start_dist) / (seg_end_dist - seg_start_dist)
                resampled[i] = polyline[seg_idx] + t * (polyline[seg_idx + 1] - polyline[seg_idx])
            else:
                resampled[i] = polyline[seg_idx]
                
        return resampled
        
    def _compute_headings(self, centerline: np.ndarray) -> np.ndarray:
        """Compute heading angles for centerline points."""
        if len(centerline) < 2:
            return np.zeros(len(centerline))
            
        headings = np.zeros(len(centerline))
        
        # Forward differences for most points
        for i in range(len(centerline) - 1):
            dx = centerline[i+1, 0] - centerline[i, 0]
            dy = centerline[i+1, 1] - centerline[i, 1]
            headings[i] = np.arctan2(dy, dx)
            
        # Use last computed heading for final point
        headings[-1] = headings[-2] if len(headings) > 1 else 0.0
        
        return headings
        
    def _lane_type_to_int(self, lane_type: LaneType) -> int:
        """Convert lane type to integer encoding."""
        type_mapping = {
            LaneType.FREEWAY: 0,
            LaneType.SURFACE_STREET: 1,
            LaneType.BIKE_LANE: 2,
            LaneType.BUS_LANE: 3,
            LaneType.PARKING: 4,
            LaneType.SHOULDER: 5
        }
        return type_mapping.get(lane_type, 0)


class LaneGraphEncoder(nn.Module):
    """
    Neural network encoder for lane graph features.
    
    Uses either GNN or attention mechanisms to encode lane graph
    structure and features into a fixed-size representation.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 3,
        encoder_type: str = "attention",  # "gnn" or "attention"
        max_lanes: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize lane graph encoder.
        
        Args:
            input_dim: Input feature dimension per lane point
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of encoder layers
            encoder_type: Type of encoder ("gnn" or "attention")
            max_lanes: Maximum number of lanes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        self.max_lanes = max_lanes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Polyline encoder (per-lane)
        self.polyline_encoder = PolylineEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            max_points=20
        )
        
        # Graph-level encoder
        if encoder_type == "gnn":
            self.graph_encoder = LaneGNNEncoder(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif encoder_type == "attention":
            self.graph_encoder = LaneAttentionEncoder(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        lane_features: torch.Tensor,
        lane_mask: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through lane graph encoder.
        
        Args:
            lane_features: Lane features [B, L, P, D]
            lane_mask: Lane point mask [B, L, P]
            adjacency: Lane adjacency matrix [B, L, L] (for GNN)
            
        Returns:
            Encoded lane graph features [B, output_dim]
        """
        batch_size, num_lanes, num_points, _ = lane_features.shape
        
        # Project input features
        lane_features = self.input_proj(lane_features)  # [B, L, P, H]
        
        # Encode each lane polyline
        lane_embeddings = []
        for i in range(num_lanes):
            lane_points = lane_features[:, i]  # [B, P, H]
            point_mask = lane_mask[:, i]  # [B, P]
            
            lane_emb = self.polyline_encoder(lane_points, point_mask)  # [B, H]
            lane_embeddings.append(lane_emb)
            
        lane_embeddings = torch.stack(lane_embeddings, dim=1)  # [B, L, H]
        
        # Create lane-level mask
        lane_level_mask = lane_mask.any(dim=2)  # [B, L]
        
        # Apply graph-level encoder
        if self.encoder_type == "gnn" and adjacency is not None:
            graph_features = self.graph_encoder(
                lane_embeddings, lane_level_mask, adjacency
            )
        else:
            graph_features = self.graph_encoder(
                lane_embeddings, lane_level_mask
            )
            
        # Global pooling
        masked_features = graph_features * lane_level_mask.unsqueeze(-1)
        pooled_features = masked_features.sum(dim=1) / (lane_level_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # Output projection
        output = self.output_proj(self.dropout(pooled_features))
        
        return output


class PolylineEncoder(nn.Module):
    """Encoder for individual lane polylines."""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        max_points: int = 20
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_points = max_points
        
        # Point-wise encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Sequence encoder (LSTM or Transformer)
        self.sequence_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self, 
        points: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode polyline points.
        
        Args:
            points: Point features [B, P, D]
            mask: Point mask [B, P]
            
        Returns:
            Polyline embedding [B, output_dim]
        """
        # Encode individual points
        point_features = self.point_encoder(points)  # [B, P, H]
        
        # Apply mask
        masked_features = point_features * mask.unsqueeze(-1)
        
        # Sequence encoding
        packed_input = nn.utils.rnn.pack_padded_sequence(
            masked_features, 
            mask.sum(dim=1).cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_output, (hidden, _) = self.sequence_encoder(packed_input)
        
        # Use final hidden state
        polyline_embedding = hidden[-1]  # [B, H]
        
        # Output projection
        output = self.output_proj(polyline_embedding)
        
        return output


class LaneGNNEncoder(nn.Module):
    """Graph Neural Network encoder for lane connectivity."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            LaneGNNLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply GNN layers.
        
        Args:
            node_features: Node features [B, N, H]
            node_mask: Node mask [B, N]
            adjacency: Adjacency matrix [B, N, N]
            
        Returns:
            Updated node features [B, N, H]
        """
        x = node_features
        
        for layer in self.gnn_layers:
            x = layer(x, node_mask, adjacency)
            
        return x


class LaneGNNLayer(nn.Module):
    """Single GNN layer for lane graph processing."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GNN layer.
        
        Args:
            x: Node features [B, N, H]
            mask: Node mask [B, N]
            adjacency: Adjacency matrix [B, N, N]
            
        Returns:
            Updated node features [B, N, H]
        """
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Compute messages
        x_expanded = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [B, N, N, H]
        x_neighbors = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [B, N, N, H]
        
        # Concatenate node and neighbor features
        edge_features = torch.cat([x_expanded, x_neighbors], dim=-1)  # [B, N, N, 2H]
        
        # Compute messages
        messages = self.message_net(edge_features)  # [B, N, N, H]
        
        # Apply adjacency mask
        adjacency_mask = adjacency.unsqueeze(-1)  # [B, N, N, 1]
        messages = messages * adjacency_mask
        
        # Aggregate messages
        aggregated = messages.sum(dim=2)  # [B, N, H]
        
        # Update nodes
        update_input = torch.cat([x, aggregated], dim=-1)  # [B, N, 2H]
        updates = self.update_net(update_input)  # [B, N, H]
        
        # Residual connection and normalization
        x_new = x + self.dropout(updates)
        x_new = self.layer_norm(x_new)
        
        # Apply node mask
        x_new = x_new * mask.unsqueeze(-1)
        
        return x_new


class LaneAttentionEncoder(nn.Module):
    """Attention-based encoder for lane graph processing."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(2 * num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention layers.
        
        Args:
            x: Node features [B, N, H]
            mask: Node mask [B, N]
            
        Returns:
            Updated node features [B, N, H]
        """
        # Convert mask to attention mask (True = masked)
        attn_mask = ~mask  # [B, N]
        
        for i in range(self.num_layers):
            # Self-attention
            residual = x
            x = self.layer_norms[2*i](x)
            
            attn_output, _ = self.attention_layers[i](
                x, x, x, key_padding_mask=attn_mask
            )
            x = residual + self.dropout(attn_output)
            
            # Feed-forward
            residual = x
            x = self.layer_norms[2*i + 1](x)
            ffn_output = self.ffn_layers[i](x)
            x = residual + self.dropout(ffn_output)
            
        # Apply final mask
        x = x * mask.unsqueeze(-1)
        
        return x


class LaneGraphBuilder:
    """Builder for constructing lane graphs from map data."""
    
    def __init__(self):
        """Initialize lane graph builder."""
        pass
        
    def build_from_dict(self, map_data: Dict) -> LaneGraph:
        """
        Build lane graph from dictionary representation.
        
        Args:
            map_data: Dictionary containing lane and connection data
            
        Returns:
            Constructed lane graph
        """
        graph = LaneGraph()
        
        # Add lanes
        lanes_data = map_data.get("lanes", [])
        for lane_data in lanes_data:
            lane = self._parse_lane_data(lane_data)
            if lane:
                graph.add_lane(lane)
                
        # Add connections
        connections_data = map_data.get("connections", [])
        for conn_data in connections_data:
            connection = self._parse_connection_data(conn_data)
            if connection:
                try:
                    graph.add_connection(connection)
                except ValueError:
                    # Skip invalid connections
                    continue
                    
        return graph
        
    def _parse_lane_data(self, lane_data: Dict) -> Optional[LaneSegment]:
        """Parse lane data from dictionary."""
        try:
            lane_id = lane_data.get("id", "")
            centerline = np.array(lane_data.get("centerline", []))
            
            if len(centerline) == 0:
                return None
                
            # Get boundaries (use centerline if not available)
            left_boundary = np.array(lane_data.get("left_boundary", centerline))
            right_boundary = np.array(lane_data.get("right_boundary", centerline))
            
            # Get other properties
            speed_limit = float(lane_data.get("speed_limit", 13.89))  # Default 50 km/h
            lane_type_str = lane_data.get("lane_type", "FREEWAY")
            width = float(lane_data.get("width", 3.5))  # Default 3.5m
            
            # Parse lane type
            try:
                lane_type = LaneType(lane_type_str)
            except ValueError:
                lane_type = LaneType.FREEWAY
                
            return LaneSegment(
                id=lane_id,
                centerline=centerline,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                speed_limit=speed_limit,
                lane_type=lane_type,
                width=width
            )
            
        except Exception:
            return None
            
    def _parse_connection_data(self, conn_data: Dict) -> Optional[LaneConnection]:
        """Parse connection data from dictionary."""
        try:
            from_lane = conn_data.get("from", "")
            to_lane = conn_data.get("to", "")
            conn_type_str = conn_data.get("type", "SUCCESSOR")
            cost = float(conn_data.get("cost", 1.0))
            
            # Parse connection type
            try:
                conn_type = ConnectionType(conn_type_str)
            except ValueError:
                conn_type = ConnectionType.SUCCESSOR
                
            return LaneConnection(
                from_lane=from_lane,
                to_lane=to_lane,
                connection_type=conn_type,
                cost=cost
            )
            
        except Exception:
            return None