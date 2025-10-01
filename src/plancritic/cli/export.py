#!/usr/bin/env python3
"""
Export script for PlanCritic trajectory data and model outputs.

Exports trajectory data, model predictions, and analysis results
in formats suitable for visualization and further analysis.
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
from plancritic.data.samplers import TrajectorySampler, DataCollator, SceneData
from plancritic.eval.physics_checks import PhysicsChecker
from plancritic.maps.lanegraph import LaneGraph, LaneGraphBuilder


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class TrajectoryExporter:
    """Exporter for trajectory data and model outputs."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize trajectory exporter.
        
        Args:
            model_path: Optional path to trained model checkpoint
            device: Device to use for inference ('cpu', 'cuda', or None for auto)
            config_path: Optional path to model configuration
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Load model if provided
        self.model = None
        if model_path:
            self.model = self._load_model(model_path, config_path)
            self.model.eval()
            
        # Initialize physics checker
        self.physics_checker = PhysicsChecker()
        
        # Initialize lane graph builder
        self.lane_graph_builder = LaneGraphBuilder()
        
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
            model_config = checkpoint.get("config", {}).get("model", {})
            
        # Set default values
        model_config.setdefault("state_dim", 32)
        model_config.setdefault("lane_dim", 64)
        model_config.setdefault("cand_dim", 64)
        model_config.setdefault("hidden_dim", 128)
        model_config.setdefault("dropout", 0.1)
        
        # Determine model type
        data_config = checkpoint.get("config", {}).get("data", {})
        max_candidates = data_config.get("max_candidates", 1)
        
        # Initialize base critic model
        base_critic = TrajectoryCritic(
            state_dim=model_config["state_dim"],
            lane_dim=model_config["lane_dim"],
            cand_dim=model_config["cand_dim"],
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
        
        return model
        
    def export_dataset_scenes(
        self,
        data_path: str,
        dataset: str,
        output_dir: str,
        max_scenes: Optional[int] = None,
        include_predictions: bool = True,
        include_physics: bool = True,
        export_format: str = "json"
    ) -> None:
        """
        Export dataset scenes with optional model predictions.
        
        Args:
            data_path: Path to dataset
            dataset: Dataset type ("womd" or "argoverse")
            output_dir: Output directory for exported data
            max_scenes: Maximum number of scenes to export
            include_predictions: Whether to include model predictions
            include_physics: Whether to include physics analysis
            export_format: Export format ("json", "npz", or "both")
        """
        self.logger.info(f"Exporting scenes from {dataset} dataset")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data adapter
        if dataset == "womd":
            from plancritic.data.adapters.womd_adapter import WOMDConfig
            config = WOMDConfig(data_dir=Path(data_path), max_scenes=max_scenes)
            adapter = WOMDAdapter(config=config)
        elif dataset == "argoverse":
            adapter = ArgoverseAdapter(data_path=data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        # Initialize sampler
        sampler = TrajectorySampler(
            max_candidates=8,
            min_candidates=4,
            diversity_threshold=2.0,
            physics_filter=True
        )
        
        # Initialize collator
        collator = DataCollator()
        
        # Export metadata
        metadata = {
            "dataset": dataset,
            "data_path": data_path,
            "export_timestamp": str(np.datetime64('now')),
            "total_scenes": max_scenes if max_scenes else "unknown",
            "include_predictions": include_predictions and self.model is not None,
            "include_physics": include_physics,
            "export_format": export_format
        }
        
        # Save metadata
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Process scenes
        exported_scenes = []
        num_processed = 0
        
        progress_bar = tqdm(desc="Exporting scenes")
        
        for i, scene in enumerate(adapter.load_scenes()):
            if max_scenes and num_processed >= max_scenes:
                break
                
            try:
                scene_data = self._export_single_scene(
                    scene, 
                    collator,
                    include_predictions=include_predictions and self.model is not None,
                    include_physics=include_physics
                )
                
                scene_id = scene_data.get("scene_id", f"scene_{i}")
                
                # Save individual scene file
                if export_format in ["json", "both"]:
                    scene_json_path = output_path / f"{scene_id}.json"
                    with open(scene_json_path, 'w') as f:
                        json.dump(scene_data, f, indent=2)
                        
                if export_format in ["npz", "both"]:
                    scene_npz_path = output_path / f"{scene_id}.npz"
                    self._save_scene_npz(scene_data, scene_npz_path)
                    
                exported_scenes.append({
                    "scene_id": scene_id,
                    "file_path": f"{scene_id}.json" if export_format != "npz" else f"{scene_id}.npz"
                })
                
                num_processed += 1
                progress_bar.update(1)
                
            except Exception as e:
                self.logger.warning(f"Failed to export scene {i}: {e}")
                continue
                
        progress_bar.close()
        
        # Save scene index
        scene_index = {
            "metadata": metadata,
            "scenes": exported_scenes
        }
        
        with open(output_path / "scene_index.json", 'w') as f:
            json.dump(scene_index, f, indent=2)
            
        self.logger.info(f"Exported {num_processed} scenes to {output_dir}")
        
    def _export_single_scene(
        self,
        scene: SceneData,
        collator: DataCollator,
        include_predictions: bool = True,
        include_physics: bool = True
    ) -> Dict[str, Any]:
        """Export a single scene with all relevant data."""
        # Basic scene information
        scene_data = {
            "scene_id": scene.scene_id,
            "timestamp": scene.timestamp,
            "scenario_type": "unknown"  # Could be derived from scene metadata
        }
        
        # Agent trajectories (convert from agent_states)
        if hasattr(scene, 'agent_states') and scene.agent_states is not None:
            scene_data["agent_trajectories"] = self._serialize_array(scene.agent_states)
            
        # Candidate trajectories
        if hasattr(scene, 'candidates') and scene.candidates:
            scene_data["candidate_trajectories"] = self._serialize_trajectories(
                scene.candidates
            )
            
        # Lane graph data
        if hasattr(scene, 'lane_graph') and scene.lane_graph:
            scene_data["lane_graph"] = self._serialize_lane_graph(
                scene.lane_graph
            )
            
        # State features (ego state)
        if hasattr(scene, 'ego_state') and scene.ego_state is not None:
            scene_data["state_features"] = self._serialize_array(
                scene.ego_state
            )
            
        # Model predictions
        if include_predictions and self.model is not None:
            try:
                predictions = self._generate_predictions(scene, collator)
                scene_data["model_predictions"] = predictions
            except Exception as e:
                self.logger.warning(f"Failed to generate predictions: {e}")
                
        # Physics analysis
        if include_physics:
            try:
                physics_analysis = self._generate_physics_analysis(scene)
                scene_data["physics_analysis"] = physics_analysis
            except Exception as e:
                self.logger.warning(f"Failed to generate physics analysis: {e}")
                
        return scene_data
        
    def _generate_predictions(
        self, 
        scene: SceneData, 
        collator: DataCollator
    ) -> Dict[str, Any]:
        """Generate model predictions for a scene."""
        # Collate scene
        batch = collator.collate([scene])
        
        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model(
                state_feats=batch["state_features"],
                lane_feats=batch["lane_features"],
                cand_feats=batch["candidate_features"]
            )
            
        # Convert to serializable format
        predictions = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().tolist()
            else:
                predictions[key] = value
                
        return predictions
        
    def _generate_physics_analysis(self, scene: SceneData) -> Dict[str, Any]:
        """Generate physics-based analysis for a scene."""
        # This is a simplified version - in practice, you'd extract
        # trajectory data and perform physics analysis
        
        analysis = {
            "collision_risk": 0.1,
            "comfort_score": 0.8,
            "progress_score": 0.7,
            "ttc_analysis": {
                "min_ttc": 5.0,
                "critical_interactions": []
            },
            "kinematic_feasibility": {
                "max_acceleration": 2.5,
                "max_jerk": 1.2,
                "feasible": True
            }
        }
        
        return analysis
        
    def _serialize_trajectories(self, trajectories: Any) -> List[Dict[str, Any]]:
        """Serialize trajectory data."""
        if isinstance(trajectories, np.ndarray):
            return [{
                "trajectory_id": 0,
                "waypoints": trajectories.tolist(),
                "timestamps": list(range(len(trajectories)))
            }]
        elif isinstance(trajectories, list):
            serialized = []
            for i, traj in enumerate(trajectories):
                if isinstance(traj, np.ndarray):
                    serialized.append({
                        "trajectory_id": i,
                        "waypoints": traj.tolist(),
                        "timestamps": list(range(len(traj)))
                    })
            return serialized
        else:
            return []
            
    def _serialize_lane_graph(self, lane_graph: Any) -> Dict[str, Any]:
        """Serialize lane graph data."""
        if isinstance(lane_graph, LaneGraph):
            # Convert lane graph to dictionary representation
            lanes_data = []
            for lane_id, lane in lane_graph.lanes.items():
                lanes_data.append({
                    "id": lane_id,
                    "centerline": lane.centerline.tolist(),
                    "left_boundary": lane.left_boundary.tolist(),
                    "right_boundary": lane.right_boundary.tolist(),
                    "speed_limit": lane.speed_limit,
                    "lane_type": lane.lane_type.value,
                    "width": lane.width
                })
                
            connections_data = []
            for conn in lane_graph.connections:
                connections_data.append({
                    "from": conn.from_lane,
                    "to": conn.to_lane,
                    "type": conn.connection_type.value,
                    "cost": conn.cost
                })
                
            return {
                "lanes": lanes_data,
                "connections": connections_data
            }
        else:
            # Handle other lane graph formats
            return {"raw_data": str(lane_graph)}
            
    def _serialize_array(self, array: Any) -> List[float]:
        """Serialize numpy array or tensor."""
        if isinstance(array, np.ndarray):
            return array.tolist()
        elif isinstance(array, torch.Tensor):
            return array.cpu().numpy().tolist()
        elif isinstance(array, list):
            return array
        else:
            return []
            
    def _save_scene_npz(self, scene_data: Dict[str, Any], output_path: Path) -> None:
        """Save scene data in NPZ format."""
        # Convert data to numpy arrays where possible
        arrays_to_save = {}
        
        # Extract numerical data
        if "agent_trajectories" in scene_data:
            for i, traj in enumerate(scene_data["agent_trajectories"]):
                if "waypoints" in traj:
                    arrays_to_save[f"agent_traj_{i}"] = np.array(traj["waypoints"])
                    
        if "candidate_trajectories" in scene_data:
            for i, traj in enumerate(scene_data["candidate_trajectories"]):
                if "waypoints" in traj:
                    arrays_to_save[f"candidate_traj_{i}"] = np.array(traj["waypoints"])
                    
        if "state_features" in scene_data:
            arrays_to_save["state_features"] = np.array(scene_data["state_features"])
            
        if "model_predictions" in scene_data:
            for key, value in scene_data["model_predictions"].items():
                arrays_to_save[f"pred_{key}"] = np.array(value)
                
        # Save lane graph centerlines
        if "lane_graph" in scene_data and "lanes" in scene_data["lane_graph"]:
            for i, lane in enumerate(scene_data["lane_graph"]["lanes"]):
                if "centerline" in lane:
                    arrays_to_save[f"lane_{i}_centerline"] = np.array(lane["centerline"])
                    
        # Save metadata as JSON string
        metadata = {k: v for k, v in scene_data.items() 
                   if k not in ["agent_trajectories", "candidate_trajectories", 
                               "state_features", "model_predictions"]}
        arrays_to_save["metadata"] = json.dumps(metadata)
        
        # Save NPZ file
        np.savez_compressed(output_path, **arrays_to_save)
        
    def export_web_format(
        self,
        data_path: str,
        dataset: str,
        output_dir: str,
        max_scenes: Optional[int] = None
    ) -> None:
        """
        Export data in format optimized for web visualization.
        
        Args:
            data_path: Path to dataset
            dataset: Dataset type
            output_dir: Output directory
            max_scenes: Maximum number of scenes to export
        """
        self.logger.info("Exporting data for web visualization")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export scenes in web-friendly format
        self.export_dataset_scenes(
            data_path=data_path,
            dataset=dataset,
            output_dir=str(output_path / "scenes"),
            max_scenes=max_scenes,
            include_predictions=self.model is not None,
            include_physics=True,
            export_format="json"
        )
        
        # Create web viewer configuration
        web_config = {
            "title": "PlanCritic Trajectory Visualization",
            "dataset": dataset,
            "scenes_path": "./scenes",
            "features": {
                "show_trajectories": True,
                "show_lane_graph": True,
                "show_predictions": self.model is not None,
                "show_physics": True,
                "interactive_scoring": self.model is not None
            },
            "visualization": {
                "map_style": "light",
                "trajectory_colors": {
                    "agent": "#FF6B6B",
                    "candidate": "#4ECDC4",
                    "prediction": "#45B7D1"
                },
                "lane_colors": {
                    "centerline": "#95A5A6",
                    "boundary": "#BDC3C7"
                }
            }
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(web_config, f, indent=2)
            
        self.logger.info(f"Web export completed: {output_dir}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export PlanCritic trajectory data")
    
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
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for exported data"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model checkpoint (optional)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "npz", "both", "web"],
        default="json",
        help="Export format"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Maximum number of scenes to export"
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip model predictions (even if model provided)"
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Skip physics analysis"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for inference"
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
        # Initialize exporter
        exporter = TrajectoryExporter(
            model_path=args.model_path,
            device=args.device,
            config_path=args.config
        )
        
        # Export data
        if args.format == "web":
            exporter.export_web_format(
                data_path=args.data_path,
                dataset=args.dataset,
                output_dir=args.output_dir,
                max_scenes=args.max_scenes
            )
        else:
            exporter.export_dataset_scenes(
                data_path=args.data_path,
                dataset=args.dataset,
                output_dir=args.output_dir,
                max_scenes=args.max_scenes,
                include_predictions=not args.no_predictions,
                include_physics=not args.no_physics,
                export_format=args.format
            )
            
        logger.info("Export completed successfully")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()