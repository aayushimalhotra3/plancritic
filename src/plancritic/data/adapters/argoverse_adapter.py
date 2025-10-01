"""
Argoverse 2 dataset adapter.

Provides lightweight loading of Argoverse 2 data for trajectory criticism.
Focuses on motion forecasting scenarios and essential features.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import pickle
from dataclasses import dataclass

from ..samplers import SceneData, TrajectoryCandidate


@dataclass
class ArgoverseConfig:
    """Configuration for Argoverse 2 data loading."""
    data_dir: Path
    split: str = "train"  # train, val, test
    max_scenes: Optional[int] = None
    cache_dir: Optional[Path] = None
    scenario_length: int = 110  # 11 seconds at 10Hz
    history_length: int = 50   # 5 seconds history
    future_length: int = 60    # 6 seconds future


class ArgoverseAdapter:
    """
    Lightweight adapter for Argoverse 2 Motion Forecasting Dataset.
    
    Loads essential trajectory and scene data without requiring
    the full Argoverse infrastructure. Focuses on metadata and
    features needed for trajectory criticism.
    """
    
    def __init__(self, config: ArgoverseConfig):
        """
        Initialize Argoverse adapter.
        
        Args:
            config: Argoverse loading configuration
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"Argoverse data directory not found: {self.data_dir}")
            
        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def load_scenes(self) -> Iterator[SceneData]:
        """
        Load scenes from Argoverse 2 dataset.
        
        Yields:
            SceneData objects for trajectory criticism
        """
        scenario_files = self._get_scenario_files()
        
        for i, scenario_file in enumerate(scenario_files):
            if self.config.max_scenes and i >= self.config.max_scenes:
                break
                
            try:
                scene_data = self._load_scenario_file(scenario_file)
                if scene_data:
                    yield scene_data
            except Exception as e:
                print(f"Warning: Failed to load scenario {scenario_file}: {e}")
                continue
                
    def _get_scenario_files(self) -> List[Path]:
        """Get list of scenario files to process."""
        # Look for Argoverse scenario files
        split_dir = self.data_dir / self.config.split
        
        if split_dir.exists():
            # Standard Argoverse structure
            scenario_files = list(split_dir.glob("*.parquet"))
            if not scenario_files:
                scenario_files = list(split_dir.glob("*.json"))
        else:
            # Fallback to any parquet/json files in data_dir
            scenario_files = list(self.data_dir.glob("*.parquet"))
            if not scenario_files:
                scenario_files = list(self.data_dir.glob("*.json"))
                
        return sorted(scenario_files)
        
    def _load_scenario_file(self, scenario_file: Path) -> Optional[SceneData]:
        """Load a single scenario file."""
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{scenario_file.stem}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass  # Cache miss, load from source
                    
        # Load scenario data
        scenario_data = None
        
        if scenario_file.suffix == '.parquet':
            scenario_data = self._load_parquet_scenario(scenario_file)
        elif scenario_file.suffix == '.json':
            scenario_data = self._load_json_scenario(scenario_file)
        else:
            print(f"Unsupported file format: {scenario_file}")
            return None
            
        if not scenario_data:
            return None
            
        # Parse scene data
        scene_data = self._parse_scenario_data(scenario_data, scenario_file.stem)
        
        # Cache if enabled
        if self.cache_dir and scene_data:
            cache_file = self.cache_dir / f"{scenario_file.stem}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(scene_data, f)
            except Exception:
                pass  # Cache write failed, continue
                
        return scene_data
        
    def _load_parquet_scenario(self, scenario_file: Path) -> Optional[Dict]:
        """Load scenario from parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(scenario_file)
            
            # Convert to dictionary format
            scenario_data = {
                "scenario_id": scenario_file.stem,
                "tracks": df.to_dict('records')
            }
            return scenario_data
            
        except ImportError:
            print("pandas required for parquet loading. Install with: pip install pandas")
            return None
        except Exception as e:
            print(f"Failed to load parquet {scenario_file}: {e}")
            return None
            
    def _load_json_scenario(self, scenario_file: Path) -> Optional[Dict]:
        """Load scenario from JSON file."""
        try:
            with open(scenario_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON {scenario_file}: {e}")
            return None
            
    def _parse_scenario_data(self, scenario_data: Dict, scenario_id: str) -> Optional[SceneData]:
        """Parse scenario data into SceneData format."""
        try:
            tracks = scenario_data.get("tracks", [])
            if not tracks:
                return None
                
            # Find focal agent (usually the one to predict)
            focal_track = self._find_focal_track(tracks)
            if not focal_track:
                return None
                
            # Extract ego state (focal agent at current timestep)
            ego_state = self._extract_ego_state(focal_track)
            
            # Extract other agent states
            agent_states, agent_mask = self._extract_agent_states(tracks, focal_track)
            
            # Extract lane graph (simplified from map data)
            lane_graph = self._extract_lane_graph(scenario_data)
            
            # Extract route (if available)
            route_waypoints = self._extract_route(scenario_data, focal_track)
            
            # Generate trajectory candidates from ground truth
            candidates = self._generate_trajectory_candidates(focal_track)
            
            # Get timestamp
            timestamp = scenario_data.get("timestamp", 0.0)
            
            return SceneData(
                ego_state=ego_state,
                lane_graph=lane_graph,
                agent_states=agent_states,
                agent_mask=agent_mask,
                route_waypoints=route_waypoints,
                candidates=candidates,
                scene_id=scenario_id,
                timestamp=timestamp
            )
            
        except Exception as e:
            print(f"Failed to parse scenario {scenario_id}: {e}")
            return None
            
    def _find_focal_track(self, tracks: List[Dict]) -> Optional[Dict]:
        """Find the focal agent track (agent to predict)."""
        # Look for track marked as focal/target
        for track in tracks:
            if track.get("object_category") == "FOCAL_TRACK":
                return track
            if track.get("track_id") == "AV":  # Autonomous vehicle
                return track
                
        # Fallback: find track with most future timesteps
        best_track = None
        max_future_steps = 0
        
        for track in tracks:
            trajectory = track.get("trajectory", [])
            future_steps = sum(1 for step in trajectory if step.get("timestep", 0) >= 50)
            
            if future_steps > max_future_steps:
                max_future_steps = future_steps
                best_track = track
                
        return best_track
        
    def _extract_ego_state(self, focal_track: Dict) -> np.ndarray:
        """Extract ego vehicle state from focal track."""
        trajectory = focal_track.get("trajectory", [])
        
        # Find current timestep (usually around timestep 50)
        current_step = None
        for step in trajectory:
            if step.get("timestep", 0) == 50:  # Current timestep
                current_step = step
                break
                
        if not current_step:
            # Fallback to middle of trajectory
            mid_idx = len(trajectory) // 2
            current_step = trajectory[mid_idx] if trajectory else {}
            
        # Extract state: [x, y, vx, vy, ax, ay, heading, yaw_rate]
        state = np.zeros(8)
        state[0] = current_step.get("position_x", 0.0)
        state[1] = current_step.get("position_y", 0.0)
        state[2] = current_step.get("velocity_x", 0.0)
        state[3] = current_step.get("velocity_y", 0.0)
        state[4] = current_step.get("acceleration_x", 0.0)
        state[5] = current_step.get("acceleration_y", 0.0)
        state[6] = current_step.get("heading", 0.0)
        state[7] = current_step.get("yaw_rate", 0.0)
        
        return state
        
    def _extract_agent_states(self, tracks: List[Dict], focal_track: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract states of other agents."""
        max_agents = 32
        agent_states = np.zeros((max_agents, 8))
        agent_mask = np.zeros(max_agents, dtype=bool)
        
        agent_idx = 0
        focal_id = focal_track.get("track_id")
        
        for track in tracks:
            if agent_idx >= max_agents:
                break
                
            # Skip focal track
            if track.get("track_id") == focal_id:
                continue
                
            # Skip non-vehicle tracks
            if track.get("object_type") not in ["VEHICLE", "vehicle"]:
                continue
                
            # Extract current state
            trajectory = track.get("trajectory", [])
            current_step = None
            
            for step in trajectory:
                if step.get("timestep", 0) == 50:
                    current_step = step
                    break
                    
            if current_step:
                agent_states[agent_idx, 0] = current_step.get("position_x", 0.0)
                agent_states[agent_idx, 1] = current_step.get("position_y", 0.0)
                agent_states[agent_idx, 2] = current_step.get("velocity_x", 0.0)
                agent_states[agent_idx, 3] = current_step.get("velocity_y", 0.0)
                agent_states[agent_idx, 4] = current_step.get("acceleration_x", 0.0)
                agent_states[agent_idx, 5] = current_step.get("acceleration_y", 0.0)
                agent_states[agent_idx, 6] = current_step.get("heading", 0.0)
                agent_states[agent_idx, 7] = current_step.get("yaw_rate", 0.0)
                
                agent_mask[agent_idx] = True
                agent_idx += 1
                
        return agent_states, agent_mask
        
    def _extract_lane_graph(self, scenario_data: Dict) -> Dict:
        """Extract lane graph from scenario data."""
        # Simplified lane graph (Argoverse has rich map data)
        map_data = scenario_data.get("map_features", {})
        
        lane_graph = {
            "lanes": [],
            "connections": [],
            "boundaries": []
        }
        
        # Extract lane segments
        lane_segments = map_data.get("lane_segments", [])
        for segment in lane_segments:
            lane_info = {
                "id": segment.get("id", ""),
                "centerline": segment.get("centerline", []),
                "speed_limit": segment.get("speed_limit_mph", 25) * 0.44704,  # Convert to m/s
                "lane_type": segment.get("lane_type", "VEHICLE")
            }
            lane_graph["lanes"].append(lane_info)
            
        return lane_graph
        
    def _extract_route(self, scenario_data: Dict, focal_track: Dict) -> np.ndarray:
        """Extract route waypoints."""
        # Try to get route from scenario data
        route_data = scenario_data.get("route", {})
        waypoints = route_data.get("waypoints", [])
        
        if waypoints:
            return np.array([[wp[0], wp[1]] for wp in waypoints])
            
        # Fallback: use focal track trajectory as route
        trajectory = focal_track.get("trajectory", [])
        route_points = []
        
        for step in trajectory:
            if step.get("timestep", 0) >= 50:  # Future points
                x = step.get("position_x", 0.0)
                y = step.get("position_y", 0.0)
                route_points.append([x, y])
                
        if route_points:
            return np.array(route_points)
        else:
            # Default straight route
            return np.array([[0.0, 0.0], [100.0, 0.0]])
            
    def _generate_trajectory_candidates(self, focal_track: Dict) -> List[TrajectoryCandidate]:
        """Generate trajectory candidates from ground truth and variations."""
        candidates = []
        trajectory = focal_track.get("trajectory", [])
        
        # Ground truth trajectory
        gt_waypoints = []
        gt_timestamps = []
        
        for step in trajectory:
            if step.get("timestep", 0) >= 50:  # Future trajectory
                waypoint = [
                    step.get("position_x", 0.0),
                    step.get("position_y", 0.0),
                    step.get("velocity_x", 0.0),
                    step.get("velocity_y", 0.0)
                ]
                gt_waypoints.append(waypoint)
                gt_timestamps.append(step.get("timestep", 0) * 0.1)
                
        if gt_waypoints:
            gt_candidate = TrajectoryCandidate(
                waypoints=np.array(gt_waypoints),
                timestamps=np.array(gt_timestamps),
                metadata={"planner_id": "ground_truth", "cost": 0.0, "feasible": True}
            )
            candidates.append(gt_candidate)
            
            # Generate variations
            candidates.extend(self._generate_trajectory_variations(gt_candidate))
            
        return candidates
        
    def _generate_trajectory_variations(self, gt_candidate: TrajectoryCandidate) -> List[TrajectoryCandidate]:
        """Generate trajectory variations for comparison."""
        variations = []
        gt_waypoints = gt_candidate.waypoints
        
        if len(gt_waypoints) == 0:
            return variations
            
        # Constant velocity variation
        cv_waypoints = gt_waypoints.copy()
        if len(cv_waypoints) > 1:
            # Use initial velocity
            initial_vx = cv_waypoints[0, 2]
            initial_vy = cv_waypoints[0, 3]
            
            for i in range(1, len(cv_waypoints)):
                dt = 0.1 * i
                cv_waypoints[i, 0] = gt_waypoints[0, 0] + initial_vx * dt
                cv_waypoints[i, 1] = gt_waypoints[0, 1] + initial_vy * dt
                cv_waypoints[i, 2] = initial_vx
                cv_waypoints[i, 3] = initial_vy
                
        cv_candidate = TrajectoryCandidate(
            waypoints=cv_waypoints,
            timestamps=gt_candidate.timestamps.copy(),
            metadata={"planner_id": "constant_velocity", "cost": 1.0, "feasible": True}
        )
        variations.append(cv_candidate)
        
        # Slower variation
        slow_waypoints = gt_waypoints.copy()
        slow_waypoints[:, 2:4] *= 0.8  # Reduce velocity by 20%
        
        slow_candidate = TrajectoryCandidate(
            waypoints=slow_waypoints,
            timestamps=gt_candidate.timestamps.copy(),
            metadata={"planner_id": "conservative", "cost": 1.5, "feasible": True}
        )
        variations.append(slow_candidate)
        
        return variations


def create_synthetic_argoverse_scene(scenario_id: str = "synthetic_av2_001") -> SceneData:
    """
    Create a synthetic Argoverse-style scene for testing.
    
    Args:
        scenario_id: Unique scenario identifier
        
    Returns:
        Synthetic scene data
    """
    # Ego vehicle state
    ego_state = np.array([0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Nearby agents
    agent_states = np.zeros((32, 8))
    agent_mask = np.zeros(32, dtype=bool)
    
    # Add synthetic agents
    agent_states[0] = [15.0, 3.7, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Adjacent lane
    agent_states[1] = [25.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Leading vehicle
    agent_states[2] = [-8.0, -3.7, 9.0, 0.0, 0.0, 0.0, 3.14, 0.0] # Oncoming
    agent_mask[:3] = True
    
    # Lane graph (urban intersection)
    lane_graph = {
        "lanes": [
            {
                "id": "lane_main",
                "centerline": [[i, 0.0] for i in range(-30, 101, 5)],
                "speed_limit": 11.11,  # 40 km/h
                "lane_type": "VEHICLE"
            },
            {
                "id": "lane_adjacent",
                "centerline": [[i, 3.7] for i in range(-30, 101, 5)],
                "speed_limit": 11.11,
                "lane_type": "VEHICLE"
            }
        ],
        "connections": [
            {"from": "lane_main", "to": "lane_adjacent", "type": "LANE_CHANGE"}
        ],
        "boundaries": []
    }
    
    # Route waypoints
    route_waypoints = np.array([[i, 0.0] for i in range(0, 101, 15)])
    
    # Trajectory candidates
    candidates = []
    
    # Straight trajectory
    straight_waypoints = np.zeros((60, 4))
    for t in range(60):
        straight_waypoints[t] = [t * 1.2, 0.0, 8.0, 0.0]
    candidates.append(TrajectoryCandidate(
        waypoints=straight_waypoints,
        timestamps=np.arange(60) * 0.1,
        metadata={"planner_id": "lane_follow", "cost": 1.0, "feasible": True}
    ))
    
    # Lane change trajectory
    lc_waypoints = np.zeros((60, 4))
    for t in range(60):
        # Smooth lane change over 3 seconds
        if t < 30:
            y_offset = 3.7 * (t / 30.0) * (1 - np.cos(np.pi * t / 30.0)) / 2
        else:
            y_offset = 3.7
        lc_waypoints[t] = [t * 1.2, y_offset, 8.0, 0.0]
    candidates.append(TrajectoryCandidate(
        waypoints=lc_waypoints,
        timestamps=np.arange(60) * 0.1,
        metadata={"planner_id": "lane_change_left", "cost": 2.5, "feasible": True}
    ))
    
    return SceneData(
        ego_state=ego_state,
        lane_graph=lane_graph,
        agent_states=agent_states,
        agent_mask=agent_mask,
        route_waypoints=route_waypoints,
        candidates=candidates,
        scene_id=scenario_id,
        timestamp=0.0
    )