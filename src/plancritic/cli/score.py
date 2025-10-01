#!/usr/bin/env python3
"""
Scoring script for PlanCritic trajectory critic model.

Scores trajectory candidates using a trained critic model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import torch
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plancritic.models.critic import TrajectoryCritic, MultiCandidateCritic
from plancritic.data.adapters import WOMDAdapter, ArgoverseAdapter
from plancritic.data.samplers import TrajectorySampler, DataCollator
from plancritic.eval.physics_checks import PhysicsChecker
from plancritic.eval.metrics import CriticEvaluator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class TrajectoryScorer:
    """Trajectory scoring using trained critic model."""
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize trajectory scorer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for inference ('cpu', 'cuda', or None for auto)
            config_path: Optional path to model configuration
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path, config_path)
        self.model.eval()
        
        # Initialize physics checker for comparison
        self.physics_checker = PhysicsChecker()
        
    def _load_model(
        self, 
        model_path: str, 
        config_path: Optional[str] = None
    ) -> Union[TrajectoryCritic, MultiCandidateCritic]:
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_config = config.get("model", {})
        else:
            # Try to get config from checkpoint
            model_config = checkpoint.get("config", {}).get("model", {})
            
        # Set default values if not found
        model_config.setdefault("state_dim", 8)
        model_config.setdefault("lane_dim", 6)
        model_config.setdefault("cand_dim", 4)
        model_config.setdefault("hidden_dim", 128)
        model_config.setdefault("dropout", 0.1)
        
        # Determine if multi-candidate model
        data_config = checkpoint.get("config", {}).get("data", {})
        max_candidates = data_config.get("max_candidates", 1)
        
        # Initialize base critic model with hardcoded dimensions to match training
        base_critic = TrajectoryCritic(
            state_dim=32,  # matches StateEncoder output_dim in training
            lane_dim=64,   # matches LaneGraphEncoder output_dim in training
            cand_dim=64,   # matches TrajectoryEncoder output_dim in training
            hidden=model_config.get("hidden", 128),
            dropout=model_config["dropout"]
        )
        
        # Initialize model
        if max_candidates > 1:
            model = MultiCandidateCritic(base_critic)
        else:
            model = base_critic
            
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        
        self.logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
        
    def score_trajectories(
        self,
        state_features: np.ndarray,
        lane_features: np.ndarray,
        candidate_trajectories: np.ndarray,
        return_components: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Score trajectory candidates.
        
        Args:
            state_features: Current state features [state_dim]
            lane_features: Lane graph features [lane_dim]
            candidate_trajectories: Candidate trajectories [N, seq_len, traj_dim]
            return_components: Whether to return individual score components
            
        Returns:
            Dictionary containing scores and optionally individual components
        """
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            state_tensor = torch.from_numpy(state_features).float().unsqueeze(0).to(self.device)
            lane_tensor = torch.from_numpy(lane_features).float().unsqueeze(0).to(self.device)
            
            # Handle multiple candidates
            if len(candidate_trajectories.shape) == 3:  # [N, seq_len, traj_dim]
                num_candidates = candidate_trajectories.shape[0]
                scores_list = []
                
                for i in range(num_candidates):
                    cand_tensor = torch.from_numpy(candidate_trajectories[i]).float().unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        state_feats=state_tensor,
                        lane_feats=lane_tensor,
                        cand_feats=cand_tensor
                    )
                    
                    scores_list.append(outputs)
                    
                # Combine scores
                combined_scores = {}
                for key in scores_list[0].keys():
                    combined_scores[key] = torch.cat([s[key] for s in scores_list], dim=0)
                    
            else:  # Single trajectory [seq_len, traj_dim]
                cand_tensor = torch.from_numpy(candidate_trajectories).float().unsqueeze(0).to(self.device)
                
                # Forward pass
                combined_scores = self.model(
                    state_feats=state_tensor,
                    lane_feats=lane_tensor,
                    cand_feats=cand_tensor
                )
                
        # Convert to numpy
        result = {}
        for key, value in combined_scores.items():
            result[key] = value.cpu().numpy()
            
        return result
        
    def score_from_data(
        self,
        data_path: str,
        dataset: str = "womd",
        output_path: Optional[str] = None,
        max_scenes: Optional[int] = None,
        include_physics: bool = True
    ) -> Dict[str, Any]:
        """
        Score trajectories from dataset.
        
        Args:
            data_path: Path to dataset
            dataset: Dataset type ("womd" or "argoverse")
            output_path: Optional path to save results
            max_scenes: Maximum number of scenes to process
            include_physics: Whether to include physics-based scores for comparison
            
        Returns:
            Dictionary containing scoring results
        """
        self.logger.info(f"Scoring trajectories from {dataset} dataset at {data_path}")
        
        # Initialize data adapter
        if dataset == "womd":
            from plancritic.data.adapters import WOMDConfig
            config = WOMDConfig(data_dir=data_path, max_scenes=max_scenes)
            adapter = WOMDAdapter(config)
        elif dataset == "argoverse":
            from plancritic.data.adapters import ArgoverseConfig
            config = ArgoverseConfig(data_dir=data_path, max_scenes=max_scenes)
            adapter = ArgoverseAdapter(config)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        # Initialize sampler
        scenes = list(adapter.load_scenes())
        
        # Initialize collator
        collator = DataCollator()
        
        # Process scenes
        results = {
            "scene_scores": [],
            "summary_stats": {},
            "config": {
                "dataset": dataset,
                "data_path": data_path,
                "max_scenes": max_scenes,
                "include_physics": include_physics
            }
        }
        
        num_processed = 0
        total_scenes = min(len(scenes), max_scenes) if max_scenes else len(scenes)
        
        progress_bar = tqdm(total=total_scenes, desc="Scoring scenes")
        
        for i, scene in enumerate(scenes):
            if max_scenes and num_processed >= max_scenes:
                break
                
            try:
                # Collate single scene
                batch = collator.collate([scene])
                
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Score with critic model
                with torch.no_grad():
                    # Extract tensors from batch
                    ego_states = batch["ego_states"]
                    agent_states = batch["agent_states"]
                    trajectories = batch["trajectories"]
                    
                    self.logger.debug(f"ego_states shape: {ego_states.shape}")
                    self.logger.debug(f"agent_states shape: {agent_states.shape}")
                    self.logger.debug(f"trajectories shape: {trajectories.shape}")
                    
                    # Flatten all tensors to 2D for feature extraction
                    # The model expects [B, feature_dim] inputs
                    
                    # For ego_states: flatten to [B, -1] and take first 32 features
                    ego_flat = ego_states.view(ego_states.shape[0], -1)
                    if ego_flat.shape[1] > 32:
                        state_feats = ego_flat[:, :32]
                    else:
                        # Pad if needed
                        state_feats = torch.zeros(ego_flat.shape[0], 32, device=self.device)
                        state_feats[:, :ego_flat.shape[1]] = ego_flat
                    
                    # For agent_states: flatten to [B, -1] and take first 64 features
                    agent_flat = agent_states.view(agent_states.shape[0], -1)
                    if agent_flat.shape[1] > 64:
                        lane_feats = agent_flat[:, :64]
                    else:
                        # Pad if needed
                        lane_feats = torch.zeros(agent_flat.shape[0], 64, device=self.device)
                        lane_feats[:, :agent_flat.shape[1]] = agent_flat
                    
                    # For trajectories: flatten to [B, -1] and take first 64 features
                    traj_flat = trajectories.view(trajectories.shape[0], -1)
                    if traj_flat.shape[1] > 64:
                        cand_feats = traj_flat[:, :64]
                    else:
                        # Pad if needed
                        cand_feats = torch.zeros(traj_flat.shape[0], 64, device=self.device)
                        cand_feats[:, :traj_flat.shape[1]] = traj_flat
                    
                    self.logger.debug(f"state_feats shape: {state_feats.shape}")
                    self.logger.debug(f"lane_feats shape: {lane_feats.shape}")
                    self.logger.debug(f"cand_feats shape: {cand_feats.shape}")
                    
                    # Forward pass through model with detailed debugging
                    self.logger.debug("About to call model forward...")
                    try:
                        critic_scores = self.model(state_feats, lane_feats, cand_feats)
                        self.logger.debug(f"critic_scores type: {type(critic_scores)}")
                        self.logger.debug(f"critic_scores keys: {critic_scores.keys() if isinstance(critic_scores, dict) else 'Not a dict'}")
                    except Exception as model_error:
                        import traceback
                        self.logger.error(f"Model forward pass failed: {str(model_error)}")
                        self.logger.error(f"Model forward traceback: {traceback.format_exc()}")
                        raise
                
                # Convert to numpy
                scene_result = {
                    "scene_id": scene.scene_id if hasattr(scene, 'scene_id') else f"scene_{i}",
                    "critic_scores": {}
                }
                
                # Convert each score component to numpy
                if isinstance(critic_scores, dict):
                    for key, value in critic_scores.items():
                        scene_result["critic_scores"][key] = value.cpu().numpy().tolist()
                else:
                    self.logger.error(f"Unexpected critic_scores type: {type(critic_scores)}")
                    scene_result["critic_scores"]["score"] = critic_scores.cpu().numpy().tolist()
                
                # Add physics scores if requested
                if include_physics:
                    physics_scores = self._compute_physics_scores(scene)
                    scene_result["physics_scores"] = physics_scores
                    
                results["scene_scores"].append(scene_result)
                num_processed += 1
                progress_bar.update(1)
                
            except Exception as e:
                import traceback
                self.logger.error(f"Detailed error in scene {i}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                self.logger.warning(f"Failed to process scene {i}: {e}")
                continue
                
        progress_bar.close()
        
        # Compute summary statistics
        results["summary_stats"] = self._compute_summary_stats(results["scene_scores"])
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, output_path)
            
        self.logger.info(f"Scored {num_processed} scenes")
        
        return results
        
    def _compute_physics_scores(self, sample: Dict[str, Any]) -> Dict[str, List[float]]:
        """Compute physics-based scores for comparison."""
        # This is a simplified version - in practice, you'd extract
        # trajectory data from the sample and use the physics checker
        
        # For now, return dummy physics scores
        num_candidates = 1  # Adjust based on actual sample structure
        
        return {
            "risk": [0.5] * num_candidates,
            "comfort": [0.5] * num_candidates,
            "progress": [0.5] * num_candidates
        }
        
    def _compute_summary_stats(self, scene_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from scene scores."""
        if not scene_scores:
            return {}
            
        # Collect all scores
        all_critic_scores = {
            "risk": [],
            "comfort": [],
            "progress": [],
            "score": []
        }
        
        all_physics_scores = {
            "risk": [],
            "comfort": [],
            "progress": []
        }
        
        for scene in scene_scores:
            # Critic scores
            critic = scene["critic_scores"]
            for key in all_critic_scores:
                if key in critic:
                    if isinstance(critic[key], list):
                        all_critic_scores[key].extend(critic[key])
                    else:
                        all_critic_scores[key].append(critic[key])
                        
            # Physics scores (if available)
            if "physics_scores" in scene:
                physics = scene["physics_scores"]
                for key in all_physics_scores:
                    if key in physics:
                        if isinstance(physics[key], list):
                            all_physics_scores[key].extend(physics[key])
                        else:
                            all_physics_scores[key].append(physics[key])
                            
        # Compute statistics
        stats = {
            "num_scenes": len(scene_scores),
            "critic_stats": {},
            "physics_stats": {}
        }
        
        # Critic statistics
        for key, values in all_critic_scores.items():
            if values:
                stats["critic_stats"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }
                
        # Physics statistics
        for key, values in all_physics_scores.items():
            if values:
                stats["physics_stats"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }
                
        return stats
        
    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save scoring results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Saved results to {output_path}")


def main():
    """Main scoring function."""
    parser = argparse.ArgumentParser(description="Score trajectories with PlanCritic")
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["womd", "argoverse"],
        default="womd",
        help="Dataset type"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results (JSON format)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Maximum number of scenes to process"
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Skip physics-based scoring for comparison"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize scorer
        scorer = TrajectoryScorer(
            model_path=args.model_path,
            device=args.device,
            config_path=args.config
        )
        
        # Score trajectories
        results = scorer.score_from_data(
            data_path=args.data_path,
            dataset=args.dataset,
            output_path=args.output,
            max_scenes=args.max_scenes,
            include_physics=not args.no_physics
        )
        
        # Print summary
        stats = results["summary_stats"]
        logger.info("Scoring Summary:")
        logger.info(f"  Processed scenes: {stats.get('num_scenes', 0)}")
        
        if "critic_stats" in stats:
            logger.info("  Critic scores:")
            for key, values in stats["critic_stats"].items():
                logger.info(f"    {key}: mean={values['mean']:.3f}, std={values['std']:.3f}")
                
        if "physics_stats" in stats:
            logger.info("  Physics scores:")
            for key, values in stats["physics_stats"].items():
                logger.info(f"    {key}: mean={values['mean']:.3f}, std={values['std']:.3f}")
                
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise


if __name__ == "__main__":
    main()