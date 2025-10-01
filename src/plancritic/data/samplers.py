"""
Data samplers for trajectory criticism.

Provides utilities for sampling trajectory candidates, scenes,
and creating training batches from AV datasets.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TrajectoryCandidate:
    """Represents a single trajectory candidate."""
    waypoints: np.ndarray  # [T, 4] - (x, y, vx, vy)
    timestamps: np.ndarray  # [T] - time stamps
    metadata: Dict  # Additional metadata (planner_id, cost, etc.)


@dataclass
class SceneData:
    """Represents a complete scene for trajectory evaluation."""
    ego_state: np.ndarray  # [8] - ego vehicle state
    lane_graph: Dict  # Lane graph data structure
    agent_states: np.ndarray  # [N, 8] - nearby agent states
    agent_mask: np.ndarray  # [N] - valid agent mask
    route_waypoints: np.ndarray  # [R, 2] - reference route
    candidates: List[TrajectoryCandidate]  # Trajectory candidates
    scene_id: str  # Unique scene identifier
    timestamp: float  # Scene timestamp


class TrajectorySampler:
    """
    Samples trajectory candidates for training and evaluation.
    
    Supports various sampling strategies including random sampling,
    physics-based filtering, and diversity-based selection.
    """
    
    def __init__(
        self,
        max_candidates: int = 16,
        min_candidates: int = 4,
        diversity_threshold: float = 2.0,  # meters
        physics_filter: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize trajectory sampler.
        
        Args:
            max_candidates: Maximum candidates to sample per scene
            min_candidates: Minimum candidates to sample per scene
            diversity_threshold: Minimum distance between candidates
            physics_filter: Whether to filter physically invalid trajectories
            random_seed: Random seed for reproducibility
        """
        self.max_candidates = max_candidates
        self.min_candidates = min_candidates
        self.diversity_threshold = diversity_threshold
        self.physics_filter = physics_filter
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
    def sample_candidates(
        self, 
        all_candidates: List[TrajectoryCandidate],
        ego_state: np.ndarray,
        lane_graph: Dict
    ) -> List[TrajectoryCandidate]:
        """
        Sample a subset of trajectory candidates.
        
        Args:
            all_candidates: All available trajectory candidates
            ego_state: Current ego vehicle state
            lane_graph: Lane graph for physics validation
            
        Returns:
            Sampled trajectory candidates
        """
        # Filter invalid trajectories
        if self.physics_filter:
            valid_candidates = self._filter_physics_invalid(all_candidates, lane_graph)
        else:
            valid_candidates = all_candidates
            
        if len(valid_candidates) < self.min_candidates:
            # If too few valid candidates, relax physics filter
            valid_candidates = all_candidates
            
        # Diversity-based sampling
        sampled = self._diversity_sampling(valid_candidates)
        
        # Ensure we have enough candidates
        if len(sampled) < self.min_candidates:
            # Random sampling to fill up
            remaining = [c for c in valid_candidates if c not in sampled]
            additional = random.sample(
                remaining, 
                min(self.min_candidates - len(sampled), len(remaining))
            )
            sampled.extend(additional)
            
        return sampled[:self.max_candidates]
        
    def _filter_physics_invalid(
        self, 
        candidates: List[TrajectoryCandidate],
        lane_graph: Dict
    ) -> List[TrajectoryCandidate]:
        """Filter out physically invalid trajectories."""
        valid_candidates = []
        
        for candidate in candidates:
            if self._is_physics_valid(candidate, lane_graph):
                valid_candidates.append(candidate)
                
        return valid_candidates
        
    def _is_physics_valid(
        self, 
        candidate: TrajectoryCandidate,
        lane_graph: Dict
    ) -> bool:
        """Check if trajectory satisfies basic physics constraints."""
        waypoints = candidate.waypoints
        
        # Check velocity limits
        velocities = np.linalg.norm(waypoints[:, 2:4], axis=1)
        if np.any(velocities > 50.0):  # 50 m/s = 180 km/h
            return False
            
        # Check acceleration limits
        if len(waypoints) > 1:
            dt = 0.1  # Assume 10Hz
            accelerations = np.diff(waypoints[:, 2:4], axis=0) / dt
            accel_magnitudes = np.linalg.norm(accelerations, axis=1)
            if np.any(accel_magnitudes > 10.0):  # 10 m/s²
                return False
                
        # Check for collisions with static obstacles (simplified)
        # In practice, this would use the lane graph for detailed checks
        positions = waypoints[:, :2]
        if self._has_static_collision(positions, lane_graph):
            return False
            
        return True
        
    def _has_static_collision(self, positions: np.ndarray, lane_graph: Dict) -> bool:
        """Check for collisions with static map elements."""
        # Simplified collision check - in practice would use detailed map data
        # For now, just check if trajectory goes too far from drivable area
        
        # This is a placeholder - real implementation would check against
        # lane boundaries, obstacles, etc. from the lane_graph
        return False
        
    def _diversity_sampling(
        self, 
        candidates: List[TrajectoryCandidate]
    ) -> List[TrajectoryCandidate]:
        """Sample diverse trajectory candidates."""
        if len(candidates) <= self.max_candidates:
            return candidates
            
        sampled = []
        remaining = candidates.copy()
        
        # Start with a random candidate
        first = random.choice(remaining)
        sampled.append(first)
        remaining.remove(first)
        
        # Greedily add diverse candidates
        while len(sampled) < self.max_candidates and remaining:
            best_candidate = None
            best_min_distance = -1
            
            for candidate in remaining:
                # Compute minimum distance to already sampled candidates
                min_distance = float('inf')
                for sampled_candidate in sampled:
                    distance = self._trajectory_distance(candidate, sampled_candidate)
                    min_distance = min(min_distance, distance)
                    
                # Select candidate with maximum minimum distance
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
                    
            if best_candidate is not None:
                sampled.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
                
        return sampled
        
    def _trajectory_distance(
        self, 
        traj1: TrajectoryCandidate, 
        traj2: TrajectoryCandidate
    ) -> float:
        """Compute distance between two trajectories."""
        # Use final position distance as a simple metric
        pos1 = traj1.waypoints[-1, :2]
        pos2 = traj2.waypoints[-1, :2]
        return np.linalg.norm(pos1 - pos2)


class SceneSampler:
    """
    Samples scenes for training and evaluation.
    
    Handles scene selection, augmentation, and batch creation
    for efficient training of trajectory critics.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        augment_scenes: bool = True,
        augmentation_noise: float = 0.1,
        balance_difficulty: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize scene sampler.
        
        Args:
            batch_size: Number of scenes per batch
            augment_scenes: Whether to apply data augmentation
            augmentation_noise: Noise level for augmentation
            balance_difficulty: Whether to balance easy/hard scenes
            random_seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.augment_scenes = augment_scenes
        self.augmentation_noise = augmentation_noise
        self.balance_difficulty = balance_difficulty
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
    def sample_batch(self, scenes: List[SceneData]) -> List[SceneData]:
        """
        Sample a batch of scenes for training.
        
        Args:
            scenes: Available scenes to sample from
            
        Returns:
            Batch of sampled and potentially augmented scenes
        """
        # Sample scenes
        if self.balance_difficulty:
            sampled_scenes = self._balanced_sampling(scenes)
        else:
            sampled_scenes = random.sample(scenes, min(self.batch_size, len(scenes)))
            
        # Apply augmentation
        if self.augment_scenes:
            sampled_scenes = [self._augment_scene(scene) for scene in sampled_scenes]
            
        return sampled_scenes
        
    def _balanced_sampling(self, scenes: List[SceneData]) -> List[SceneData]:
        """Sample scenes with balanced difficulty distribution."""
        # Compute difficulty scores (placeholder implementation)
        difficulties = [self._compute_difficulty(scene) for scene in scenes]
        
        # Sort by difficulty
        sorted_indices = np.argsort(difficulties)
        
        # Sample from different difficulty ranges
        n_easy = self.batch_size // 3
        n_medium = self.batch_size // 3
        n_hard = self.batch_size - n_easy - n_medium
        
        easy_indices = sorted_indices[:len(scenes)//3]
        medium_indices = sorted_indices[len(scenes)//3:2*len(scenes)//3]
        hard_indices = sorted_indices[2*len(scenes)//3:]
        
        sampled_indices = []
        sampled_indices.extend(np.random.choice(easy_indices, n_easy, replace=False))
        sampled_indices.extend(np.random.choice(medium_indices, n_medium, replace=False))
        sampled_indices.extend(np.random.choice(hard_indices, n_hard, replace=False))
        
        return [scenes[i] for i in sampled_indices]
        
    def _compute_difficulty(self, scene: SceneData) -> float:
        """Compute scene difficulty score."""
        # Simple heuristic based on number of agents and scene complexity
        num_agents = np.sum(scene.agent_mask)
        agent_velocities = np.linalg.norm(scene.agent_states[:, 2:4], axis=1)
        avg_velocity = np.mean(agent_velocities[scene.agent_mask])
        
        # Higher difficulty for more agents and higher velocities
        difficulty = num_agents * 0.1 + avg_velocity * 0.01
        
        return difficulty
        
    def _augment_scene(self, scene: SceneData) -> SceneData:
        """Apply data augmentation to a scene."""
        # Create a copy to avoid modifying original
        augmented_scene = SceneData(
            ego_state=scene.ego_state.copy(),
            lane_graph=scene.lane_graph,  # Lane graph typically not augmented
            agent_states=scene.agent_states.copy(),
            agent_mask=scene.agent_mask.copy(),
            route_waypoints=scene.route_waypoints.copy(),
            candidates=[self._augment_trajectory(c) for c in scene.candidates],
            scene_id=scene.scene_id + "_aug",
            timestamp=scene.timestamp
        )
        
        # Add noise to positions and velocities
        noise_scale = self.augmentation_noise
        
        # Ego state noise
        augmented_scene.ego_state[:2] += np.random.normal(0, noise_scale, 2)  # position
        augmented_scene.ego_state[2:4] += np.random.normal(0, noise_scale*0.1, 2)  # velocity
        
        # Agent state noise
        valid_agents = augmented_scene.agent_mask
        augmented_scene.agent_states[valid_agents, :2] += np.random.normal(
            0, noise_scale, (np.sum(valid_agents), 2)
        )
        augmented_scene.agent_states[valid_agents, 2:4] += np.random.normal(
            0, noise_scale*0.1, (np.sum(valid_agents), 2)
        )
        
        return augmented_scene
        
    def _augment_trajectory(self, candidate: TrajectoryCandidate) -> TrajectoryCandidate:
        """Apply augmentation to a trajectory candidate."""
        augmented_waypoints = candidate.waypoints.copy()
        
        # Add small amount of noise to trajectory
        noise_scale = self.augmentation_noise * 0.5  # Smaller noise for trajectories
        augmented_waypoints[:, :2] += np.random.normal(
            0, noise_scale, augmented_waypoints[:, :2].shape
        )
        augmented_waypoints[:, 2:4] += np.random.normal(
            0, noise_scale*0.1, augmented_waypoints[:, 2:4].shape
        )
        
        return TrajectoryCandidate(
            waypoints=augmented_waypoints,
            timestamps=candidate.timestamps.copy(),
            metadata=candidate.metadata.copy()
        )


class DataCollator:
    """
    Collates scene data into batches for neural network training.
    
    Handles padding, tensor conversion, and batch organization.
    """
    
    def __init__(
        self,
        max_agents: int = 32,
        max_candidates: int = 16,
        trajectory_length: int = 80,
        device: str = "cpu"
    ):
        """
        Initialize data collator.
        
        Args:
            max_agents: Maximum number of agents per scene
            max_candidates: Maximum number of trajectory candidates
            trajectory_length: Fixed trajectory length
            device: Target device for tensors
        """
        self.max_agents = max_agents
        self.max_candidates = max_candidates
        self.trajectory_length = trajectory_length
        self.device = device
        
    def collate(self, scenes: List[SceneData]) -> Dict[str, torch.Tensor]:
        """
        Collate scenes into a batch.
        
        Args:
            scenes: List of scene data
            
        Returns:
            Dictionary of batched tensors
        """
        batch_size = len(scenes)
        
        # Initialize batch tensors
        ego_states = torch.zeros(batch_size, 8, device=self.device)
        agent_states = torch.zeros(batch_size, self.max_agents, 8, device=self.device)
        agent_masks = torch.zeros(batch_size, self.max_agents, dtype=torch.bool, device=self.device)
        
        trajectories = torch.zeros(
            batch_size, self.max_candidates, self.trajectory_length, 4, device=self.device
        )
        trajectory_masks = torch.zeros(
            batch_size, self.max_candidates, dtype=torch.bool, device=self.device
        )
        
        # Fill batch tensors
        for i, scene in enumerate(scenes):
            # Ego state
            ego_states[i] = torch.from_numpy(scene.ego_state).float()
            
            # Agent states
            n_agents = min(len(scene.agent_states), self.max_agents)
            agent_states[i, :n_agents] = torch.from_numpy(scene.agent_states[:n_agents]).float()
            agent_masks[i, :n_agents] = torch.from_numpy(scene.agent_mask[:n_agents])
            
            # Trajectory candidates
            n_candidates = min(len(scene.candidates), self.max_candidates)
            for j, candidate in enumerate(scene.candidates[:n_candidates]):
                traj_len = min(len(candidate.waypoints), self.trajectory_length)
                trajectories[i, j, :traj_len] = torch.from_numpy(
                    candidate.waypoints[:traj_len]
                ).float()
                trajectory_masks[i, j] = True
                
        return {
            "ego_states": ego_states,
            "agent_states": agent_states,
            "agent_masks": agent_masks,
            "trajectories": trajectories,
            "trajectory_masks": trajectory_masks,
            "batch_size": batch_size
        }