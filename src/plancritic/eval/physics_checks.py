"""
Physics-based trajectory evaluation and pseudo-label generation.

Provides functions to compute collision risk, comfort metrics, and progress
scores from trajectory data without requiring human annotations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import warnings

from ..data.samplers import TrajectoryCandidate, SceneData


@dataclass
class PhysicsConfig:
    """Configuration for physics-based evaluation."""
    # Time-to-collision parameters
    ttc_threshold: float = 3.0  # seconds
    ttc_critical: float = 1.5   # seconds for critical risk
    
    # Jerk parameters
    jerk_threshold: float = 2.0  # m/s³
    jerk_critical: float = 4.0   # m/s³ for high discomfort
    
    # Collision detection
    vehicle_length: float = 4.5  # meters
    vehicle_width: float = 2.0   # meters
    safety_margin: float = 0.5   # meters
    
    # Progress parameters
    route_tolerance: float = 5.0  # meters from route
    min_progress_speed: float = 1.0  # m/s minimum for progress
    
    # Temporal parameters
    dt: float = 0.1  # timestep in seconds


class PhysicsChecker:
    """
    Physics-based trajectory evaluation for pseudo-label generation.
    
    Computes risk, comfort, and progress scores from trajectory physics
    without requiring human annotations.
    """
    
    def __init__(self, config: PhysicsConfig = None):
        """
        Initialize physics checker.
        
        Args:
            config: Physics evaluation configuration
        """
        self.config = config or PhysicsConfig()
        
    def evaluate_trajectory(
        self, 
        candidate: TrajectoryCandidate, 
        scene: SceneData
    ) -> Dict[str, float]:
        """
        Evaluate a single trajectory candidate.
        
        Args:
            candidate: Trajectory to evaluate
            scene: Scene context (ego state, agents, map)
            
        Returns:
            Dictionary with risk, comfort, progress, and composite scores
        """
        # Compute individual metrics
        risk_score = self.compute_collision_risk(candidate, scene)
        comfort_score = self.compute_comfort_score(candidate)
        progress_score = self.compute_progress_score(candidate, scene)
        
        # Compute composite score
        composite_score = self._compute_composite_score(
            risk_score, comfort_score, progress_score
        )
        
        return {
            "risk": risk_score,
            "comfort": comfort_score, 
            "progress": progress_score,
            "composite": composite_score
        }
        
    def evaluate_batch(
        self,
        candidates: List[TrajectoryCandidate],
        scenes: List[SceneData]
    ) -> List[Dict[str, float]]:
        """
        Evaluate a batch of trajectory candidates.
        
        Args:
            candidates: List of trajectories to evaluate
            scenes: List of scene contexts
            
        Returns:
            List of evaluation dictionaries
        """
        if len(candidates) != len(scenes):
            raise ValueError("Number of candidates must match number of scenes")
            
        results = []
        for candidate, scene in zip(candidates, scenes):
            try:
                result = self.evaluate_trajectory(candidate, scene)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to evaluate trajectory: {e}")
                # Return default scores for failed evaluations
                results.append({
                    "risk": 0.5,
                    "comfort": 0.5,
                    "progress": 0.5,
                    "composite": 0.5
                })
                
        return results
        
    def compute_collision_risk(
        self, 
        candidate: TrajectoryCandidate, 
        scene: SceneData
    ) -> float:
        """
        Compute collision risk score based on TTC and spatial overlap.
        
        Args:
            candidate: Trajectory to evaluate
            scene: Scene with agent states
            
        Returns:
            Risk score between 0 (safe) and 1 (high risk)
        """
        waypoints = candidate.waypoints
        if len(waypoints) == 0:
            return 1.0  # No trajectory is risky
            
        # Get agent positions and velocities
        agent_states = scene.agent_states[scene.agent_mask]
        if len(agent_states) == 0:
            return 0.0  # No agents, no collision risk
            
        min_ttc = float('inf')
        min_distance = float('inf')
        
        for t, waypoint in enumerate(waypoints):
            ego_pos = waypoint[:2]  # x, y
            ego_vel = waypoint[2:4] if waypoint.shape[0] >= 4 else np.array([0.0, 0.0])
            
            # Project agent positions to this timestep
            future_time = t * self.config.dt
            
            for agent_state in agent_states:
                agent_pos = agent_state[:2]
                agent_vel = agent_state[2:4]
                
                # Project agent position
                projected_agent_pos = agent_pos + agent_vel * future_time
                
                # Compute distance
                distance = np.linalg.norm(ego_pos - projected_agent_pos)
                min_distance = min(min_distance, distance)
                
                # Compute TTC if vehicles are approaching
                relative_pos = projected_agent_pos - ego_pos
                relative_vel = agent_vel - ego_vel
                
                # Check if approaching (dot product < 0)
                if np.dot(relative_pos, relative_vel) < 0:
                    relative_speed = np.linalg.norm(relative_vel)
                    if relative_speed > 0.1:  # Avoid division by zero
                        ttc = distance / relative_speed
                        min_ttc = min(min_ttc, ttc)
                        
        # Convert to risk score
        risk_from_ttc = 0.0
        if min_ttc < self.config.ttc_threshold:
            if min_ttc < self.config.ttc_critical:
                risk_from_ttc = 1.0
            else:
                # Sigmoid mapping
                risk_from_ttc = 1.0 / (1.0 + np.exp(2.0 * (min_ttc - self.config.ttc_critical)))
                
        # Risk from minimum distance
        safety_distance = self.config.vehicle_length + self.config.safety_margin
        risk_from_distance = 0.0
        if min_distance < safety_distance:
            risk_from_distance = 1.0 - (min_distance / safety_distance)
            
        # Combine risks (take maximum)
        total_risk = max(risk_from_ttc, risk_from_distance)
        
        return float(np.clip(total_risk, 0.0, 1.0))
        
    def compute_comfort_score(self, candidate: TrajectoryCandidate) -> float:
        """
        Compute comfort score based on jerk and acceleration.
        
        Args:
            candidate: Trajectory to evaluate
            
        Returns:
            Comfort score between 0 (uncomfortable) and 1 (comfortable)
        """
        waypoints = candidate.waypoints
        if len(waypoints) < 3:
            return 1.0  # Too short to compute jerk
            
        # Compute accelerations
        velocities = waypoints[:, 2:4]  # vx, vy
        accelerations = np.diff(velocities, axis=0) / self.config.dt
        
        if len(accelerations) < 2:
            return 1.0
            
        # Compute jerk (derivative of acceleration)
        jerks = np.diff(accelerations, axis=0) / self.config.dt
        
        # Compute jerk magnitudes
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        
        # Compute comfort metrics
        max_jerk = np.max(jerk_magnitudes)
        mean_jerk = np.mean(jerk_magnitudes)
        
        # Convert to comfort score (lower jerk = higher comfort)
        comfort_from_max = 1.0
        if max_jerk > self.config.jerk_threshold:
            if max_jerk > self.config.jerk_critical:
                comfort_from_max = 0.0
            else:
                # Linear mapping
                comfort_from_max = 1.0 - (max_jerk - self.config.jerk_threshold) / \
                                  (self.config.jerk_critical - self.config.jerk_threshold)
                                  
        comfort_from_mean = 1.0
        if mean_jerk > self.config.jerk_threshold * 0.5:
            comfort_from_mean = 1.0 - (mean_jerk / (self.config.jerk_threshold * 0.5))
            
        # Combine comfort scores (weighted average)
        total_comfort = 0.7 * comfort_from_max + 0.3 * comfort_from_mean
        
        return float(np.clip(total_comfort, 0.0, 1.0))
        
    def compute_progress_score(
        self, 
        candidate: TrajectoryCandidate, 
        scene: SceneData
    ) -> float:
        """
        Compute progress score based on route following and forward motion.
        
        Args:
            candidate: Trajectory to evaluate
            scene: Scene with route information
            
        Returns:
            Progress score between 0 (no progress) and 1 (good progress)
        """
        waypoints = candidate.waypoints
        route_waypoints = scene.route_waypoints
        
        if len(waypoints) == 0:
            return 0.0
            
        if len(route_waypoints) == 0:
            # No route available, use forward motion as proxy
            return self._compute_forward_progress(waypoints)
            
        # Compute route following score
        route_following_score = self._compute_route_following(waypoints, route_waypoints)
        
        # Compute speed consistency score
        speed_score = self._compute_speed_consistency(waypoints)
        
        # Combine scores
        progress_score = 0.7 * route_following_score + 0.3 * speed_score
        
        return float(np.clip(progress_score, 0.0, 1.0))
        
    def _compute_composite_score(
        self, 
        risk: float, 
        comfort: float, 
        progress: float
    ) -> float:
        """
        Compute composite score from individual metrics.
        
        Args:
            risk: Risk score (0=safe, 1=risky)
            comfort: Comfort score (0=uncomfortable, 1=comfortable)  
            progress: Progress score (0=no progress, 1=good progress)
            
        Returns:
            Composite score (higher is better)
        """
        # Convert risk to safety (invert)
        safety = 1.0 - risk
        
        # Weighted combination (safety is most important)
        composite = 0.5 * safety + 0.2 * comfort + 0.3 * progress
        
        return float(np.clip(composite, 0.0, 1.0))
        
    def _compute_forward_progress(self, waypoints: np.ndarray) -> float:
        """Compute progress based on forward motion."""
        if len(waypoints) < 2:
            return 0.0
            
        # Compute total distance traveled
        positions = waypoints[:, :2]
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # Compute time duration
        duration = len(waypoints) * self.config.dt
        
        # Average speed
        avg_speed = total_distance / duration if duration > 0 else 0.0
        
        # Progress score based on speed
        if avg_speed < self.config.min_progress_speed:
            return avg_speed / self.config.min_progress_speed
        else:
            return 1.0
            
    def _compute_route_following(
        self, 
        waypoints: np.ndarray, 
        route_waypoints: np.ndarray
    ) -> float:
        """Compute how well trajectory follows the route."""
        positions = waypoints[:, :2]
        
        # Compute distances to route
        distances_to_route = []
        
        for pos in positions:
            # Find closest point on route
            if len(route_waypoints) == 1:
                dist = np.linalg.norm(pos - route_waypoints[0])
            else:
                # Distance to route segments
                min_dist = float('inf')
                for i in range(len(route_waypoints) - 1):
                    seg_start = route_waypoints[i]
                    seg_end = route_waypoints[i + 1]
                    dist = self._point_to_segment_distance(pos, seg_start, seg_end)
                    min_dist = min(min_dist, dist)
                dist = min_dist
                
            distances_to_route.append(dist)
            
        # Convert to score
        avg_distance = np.mean(distances_to_route)
        
        if avg_distance > self.config.route_tolerance:
            return 0.0
        else:
            return 1.0 - (avg_distance / self.config.route_tolerance)
            
    def _compute_speed_consistency(self, waypoints: np.ndarray) -> float:
        """Compute speed consistency score."""
        if len(waypoints) < 2:
            return 1.0
            
        velocities = waypoints[:, 2:4]
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Penalize very low speeds
        min_speed_penalty = np.mean(speeds < self.config.min_progress_speed)
        
        # Reward consistent speeds
        speed_std = np.std(speeds)
        consistency_score = 1.0 / (1.0 + speed_std)
        
        return (1.0 - min_speed_penalty) * consistency_score
        
    def _point_to_segment_distance(
        self, 
        point: np.ndarray, 
        seg_start: np.ndarray, 
        seg_end: np.ndarray
    ) -> float:
        """Compute distance from point to line segment."""
        # Vector from start to end
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq < 1e-6:  # Degenerate segment
            return np.linalg.norm(point - seg_start)
            
        # Project point onto segment
        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)  # Clamp to segment
        
        # Closest point on segment
        closest = seg_start + t * seg_vec
        
        return np.linalg.norm(point - closest)


def compute_ttc(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray, 
    agent_pos: np.ndarray,
    agent_vel: np.ndarray
) -> float:
    """
    Compute time-to-collision between ego and agent.
    
    Args:
        ego_pos: Ego position [x, y]
        ego_vel: Ego velocity [vx, vy]
        agent_pos: Agent position [x, y]
        agent_vel: Agent velocity [vx, vy]
        
    Returns:
        Time to collision in seconds (inf if no collision)
    """
    relative_pos = agent_pos - ego_pos
    relative_vel = agent_vel - ego_vel
    
    # Check if approaching
    if np.dot(relative_pos, relative_vel) >= 0:
        return float('inf')
        
    # Compute TTC
    relative_speed = np.linalg.norm(relative_vel)
    if relative_speed < 1e-6:
        return float('inf')
        
    distance = np.linalg.norm(relative_pos)
    return distance / relative_speed


def compute_jerk(trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Compute jerk (derivative of acceleration) from trajectory.
    
    Args:
        trajectory: Trajectory waypoints [T, 4] with [x, y, vx, vy]
        dt: Time step in seconds
        
    Returns:
        Jerk magnitudes [T-2]
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


def check_map_collision(
    trajectory: np.ndarray,
    lane_graph: Dict,
    vehicle_width: float = 2.0,
    safety_margin: float = 0.5
) -> List[bool]:
    """
    Check for collisions with static map elements.
    
    Args:
        trajectory: Trajectory waypoints [T, 4] with [x, y, vx, vy]
        lane_graph: Lane graph with boundaries
        vehicle_width: Vehicle width in meters
        safety_margin: Additional safety margin
        
    Returns:
        List of collision flags for each waypoint
    """
    positions = trajectory[:, :2]
    collisions = []
    
    # Get map boundaries
    boundaries = lane_graph.get("boundaries", [])
    
    for pos in positions:
        collision = False
        
        # Check against each boundary
        for boundary in boundaries:
            boundary_points = np.array(boundary.get("points", []))
            if len(boundary_points) < 2:
                continue
                
            # Check distance to boundary
            for i in range(len(boundary_points) - 1):
                seg_start = boundary_points[i]
                seg_end = boundary_points[i + 1]
                
                # Distance from vehicle center to boundary segment
                dist = _point_to_segment_distance(pos, seg_start, seg_end)
                
                # Check if vehicle (with margin) intersects boundary
                if dist < (vehicle_width / 2 + safety_margin):
                    collision = True
                    break
                    
            if collision:
                break
                
        collisions.append(collision)
        
    return collisions


def _point_to_segment_distance(
    point: np.ndarray, 
    seg_start: np.ndarray, 
    seg_end: np.ndarray
) -> float:
    """Helper function for point-to-segment distance."""
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq < 1e-6:
        return np.linalg.norm(point - seg_start)
        
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    
    closest = seg_start + t * seg_vec
    return np.linalg.norm(point - closest)