"""
Waymo Open Motion Dataset (WOMD) adapter.

Provides lightweight loading of WOMD data for trajectory criticism.
Focuses on metadata and essential features rather than full dataset loading.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import pickle
from dataclasses import dataclass

from ..samplers import SceneData, TrajectoryCandidate


@dataclass
class WOMDConfig:
    """Configuration for WOMD data loading."""
    data_dir: Path
    split: str = "training"  # training, validation, testing
    max_scenes: Optional[int] = None
    cache_dir: Optional[Path] = None
    load_full_trajectories: bool = False
    trajectory_length: int = 80  # 8 seconds at 10Hz


class WOMDAdapter:
    """
    Lightweight adapter for Waymo Open Motion Dataset.
    
    Loads essential trajectory and scene data without requiring
    the full WOMD infrastructure. Focuses on metadata and
    features needed for trajectory criticism.
    """
    
    def __init__(self, config: WOMDConfig):
        """
        Initialize WOMD adapter.
        
        Args:
            config: WOMD loading configuration
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"WOMD data directory not found: {self.data_dir}")
            
        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def load_scenes(self) -> Iterator[SceneData]:
        """
        Load scenes from WOMD dataset.
        
        Yields:
            SceneData objects for trajectory criticism
        """
        scene_files = self._get_scene_files()
        
        for i, scene_file in enumerate(scene_files):
            if self.config.max_scenes and i >= self.config.max_scenes:
                break
                
            try:
                scene_data = self._load_scene_file(scene_file)
                if scene_data:
                    yield scene_data
            except Exception as e:
                print(f"Warning: Failed to load scene {scene_file}: {e}")
                continue
                
    def _get_scene_files(self) -> List[Path]:
        """Get list of scene files to process."""
        # Look for preprocessed scene files
        pattern = f"*_{self.config.split}_*.json"
        scene_files = list(self.data_dir.glob(pattern))
        
        if not scene_files:
            # Fallback to any JSON files
            scene_files = list(self.data_dir.glob("*.json"))
            
        return sorted(scene_files)
        
    def _load_scene_file(self, scene_file: Path) -> Optional[SceneData]:
        """Load a single scene file."""
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{scene_file.stem}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass  # Cache miss, load from source
                    
        # Load from JSON
        try:
            with open(scene_file, 'r') as f:
                scene_json = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON {scene_file}: {e}")
            return None
            
        # Parse scene data
        scene_data = self._parse_scene_json(scene_json, scene_file.stem)
        
        # Cache if enabled
        if self.cache_dir and scene_data:
            cache_file = self.cache_dir / f"{scene_file.stem}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(scene_data, f)
            except Exception:
                pass  # Cache write failed, continue
                
        return scene_data
        
    def _parse_scene_json(self, scene_json: Dict, scene_id: str) -> Optional[SceneData]:
        """Parse scene JSON into SceneData format."""
        try:
            # Extract ego state
            ego_data = scene_json.get("ego_vehicle", {})
            ego_state = self._parse_vehicle_state(ego_data)
            
            # Extract agent states
            agents_data = scene_json.get("other_agents", [])
            agent_states, agent_mask = self._parse_agent_states(agents_data)
            
            # Extract lane graph (simplified)
            lane_graph = self._parse_lane_graph(scene_json.get("map_data", {}))
            
            # Extract route waypoints
            route_data = scene_json.get("route", {})
            route_waypoints = self._parse_route(route_data)
            
            # Extract trajectory candidates
            candidates_data = scene_json.get("trajectory_candidates", [])
            candidates = self._parse_trajectory_candidates(candidates_data)
            
            # Get timestamp
            timestamp = scene_json.get("timestamp", 0.0)
            
            return SceneData(
                ego_state=ego_state,
                lane_graph=lane_graph,
                agent_states=agent_states,
                agent_mask=agent_mask,
                route_waypoints=route_waypoints,
                candidates=candidates,
                scene_id=scene_id,
                timestamp=timestamp
            )
            
        except Exception as e:
            print(f"Failed to parse scene {scene_id}: {e}")
            return None
            
    def _parse_vehicle_state(self, vehicle_data: Dict) -> np.ndarray:
        """Parse vehicle state into standard format."""
        # Standard format: [x, y, vx, vy, ax, ay, heading, yaw_rate]
        state = np.zeros(8)
        
        # Position
        state[0] = vehicle_data.get("x", 0.0)
        state[1] = vehicle_data.get("y", 0.0)
        
        # Velocity
        state[2] = vehicle_data.get("vx", 0.0)
        state[3] = vehicle_data.get("vy", 0.0)
        
        # Acceleration
        state[4] = vehicle_data.get("acceleration", 0.0)
        state[5] = 0.0  # ay not provided in our format
        
        # Heading and yaw rate
        state[6] = vehicle_data.get("heading", 0.0)
        state[7] = vehicle_data.get("yaw_rate", 0.0)
        
        return state
        
    def _parse_agent_states(self, agents_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Parse agent states into arrays."""
        max_agents = 32  # Fixed maximum for consistency
        
        agent_states = np.zeros((max_agents, 8))
        agent_mask = np.zeros(max_agents, dtype=bool)
        
        for i, agent_data in enumerate(agents_data[:max_agents]):
            agent_states[i] = self._parse_vehicle_state(agent_data)
            agent_mask[i] = True
            
        return agent_states, agent_mask
        
    def _parse_lane_graph(self, map_data: Dict) -> Dict:
        """Parse lane graph data (simplified)."""
        # Simplified lane graph representation
        lane_graph = {
            "lanes": [],
            "connections": [],
            "boundaries": []
        }
        
        # Parse lane centerlines
        lanes = map_data.get("lanes", [])
        for lane_data in lanes:
            lane_info = {
                "id": lane_data.get("id", ""),
                "centerline": lane_data.get("centerline", []),
                "speed_limit": lane_data.get("speed_limit", 13.89),  # 50 km/h default
                "lane_type": lane_data.get("type", "FREEWAY")
            }
            lane_graph["lanes"].append(lane_info)
            
        # Parse connections (simplified)
        connections = map_data.get("connections", [])
        lane_graph["connections"] = connections
        
        return lane_graph
        
    def _parse_route(self, route_data: Dict) -> np.ndarray:
        """Parse route waypoints."""
        waypoints = route_data.get("waypoints", [])
        
        if not waypoints:
            # Default route (straight ahead)
            return np.array([[0.0, 0.0], [100.0, 0.0]])
            
        # Convert to numpy array
        route_waypoints = np.array([[wp[0], wp[1]] for wp in waypoints])
        
        return route_waypoints
        
    def _parse_trajectory_candidates(self, candidates_data: List[Dict]) -> List[TrajectoryCandidate]:
        """Parse trajectory candidates."""
        candidates = []
        
        for i, cand_data in enumerate(candidates_data):
            # Extract waypoints
            waypoints_data = cand_data.get("waypoints", [])
            if not waypoints_data:
                continue
                
            # Convert to standard format [T, 4] - (x, y, vx, vy)
            waypoints = np.zeros((len(waypoints_data), 4))
            timestamps = np.zeros(len(waypoints_data))
            
            for j, wp in enumerate(waypoints_data):
                waypoints[j, 0] = wp.get("x", 0.0)
                waypoints[j, 1] = wp.get("y", 0.0)
                waypoints[j, 2] = wp.get("vx", 0.0)
                waypoints[j, 3] = wp.get("vy", 0.0)
                timestamps[j] = wp.get("timestamp", j * 0.1)
                
            # Metadata
            metadata = {
                "planner_id": cand_data.get("planner_id", f"planner_{i}"),
                "cost": cand_data.get("cost", 0.0),
                "feasible": cand_data.get("feasible", True)
            }
            
            candidate = TrajectoryCandidate(
                waypoints=waypoints,
                timestamps=timestamps,
                metadata=metadata
            )
            candidates.append(candidate)
            
        return candidates


def create_synthetic_womd_scene(scene_id: str = "synthetic_001") -> SceneData:
    """
    Create a synthetic WOMD-style scene for testing.
    
    Args:
        scene_id: Unique scene identifier
        
    Returns:
        Synthetic scene data
    """
    # Ego vehicle state
    ego_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Nearby agents
    agent_states = np.zeros((32, 8))
    agent_mask = np.zeros(32, dtype=bool)
    
    # Add a few synthetic agents
    agent_states[0] = [20.0, 3.5, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Leading vehicle
    agent_states[1] = [-10.0, -3.5, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Oncoming vehicle
    agent_mask[:2] = True
    
    # Simple lane graph
    lane_graph = {
        "lanes": [
            {
                "id": "lane_0",
                "centerline": [[i, 0.0] for i in range(-50, 101, 10)],
                "speed_limit": 13.89,
                "lane_type": "FREEWAY"
            }
        ],
        "connections": [],
        "boundaries": []
    }
    
    # Route waypoints
    route_waypoints = np.array([[i, 0.0] for i in range(0, 101, 20)])
    
    # Trajectory candidates
    candidates = []
    
    # Straight trajectory
    straight_waypoints = np.zeros((80, 4))
    for t in range(80):
        straight_waypoints[t] = [t * 1.0, 0.0, 10.0, 0.0]
    candidates.append(TrajectoryCandidate(
        waypoints=straight_waypoints,
        timestamps=np.arange(80) * 0.1,
        metadata={"planner_id": "straight", "cost": 1.0, "feasible": True}
    ))
    
    # Lane change trajectory
    lc_waypoints = np.zeros((80, 4))
    for t in range(80):
        y_offset = 3.5 * (1 - np.exp(-t / 20.0))  # Smooth lane change
        lc_waypoints[t] = [t * 1.0, y_offset, 10.0, 0.0]
    candidates.append(TrajectoryCandidate(
        waypoints=lc_waypoints,
        timestamps=np.arange(80) * 0.1,
        metadata={"planner_id": "lane_change", "cost": 2.0, "feasible": True}
    ))
    
    return SceneData(
        ego_state=ego_state,
        lane_graph=lane_graph,
        agent_states=agent_states,
        agent_mask=agent_mask,
        route_waypoints=route_waypoints,
        candidates=candidates,
        scene_id=scene_id,
        timestamp=0.0
    )