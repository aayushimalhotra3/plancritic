#!/usr/bin/env python3
"""
Training script for PlanCritic trajectory critic model.

Trains the critic using physics-based pseudo-labels from trajectory analysis.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plancritic.models.critic import TrajectoryCritic, MultiCandidateCritic
from plancritic.models.encoders import StateEncoder, LaneGraphEncoder, TrajectoryEncoder
from plancritic.models.losses import CriticLoss, PhysicsLoss
from plancritic.data.samplers import TrajectorySampler, DataCollator, SceneData, TrajectoryCandidate
from plancritic.data.synthetic import SyntheticScenarioGenerator
from plancritic.eval.physics_checks import PhysicsChecker
from plancritic.eval.metrics import CriticEvaluator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


class TrainingConfig:
    """Training configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize training config."""
        # Default configuration
        self.model = {
            "state_dim": 32,
            "lane_dim": 64,
            "cand_dim": 64,
            "hidden_dim": 128,
            "encoder_type": "attention",
            "dropout": 0.1
        }
        
        self.training = {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "num_epochs": 50,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "warmup_steps": 1000,
            "eval_interval": 5,
            "save_interval": 10
        }
        
        self.data = {
            "dataset": "synthetic",  # "synthetic", "womd", or "argoverse"
            "data_path": "./data",
            "num_workers": 4,
            "max_candidates": 8,
            "sequence_length": 80,
            "prediction_horizon": 80,
            "synthetic_seed": 42,
            "synthetic_num_train": 400,
            "synthetic_num_val": 80,
        }
        
        self.physics = {
            "collision_threshold": 2.0,
            "comfort_threshold": 4.0,
            "ttc_threshold": 3.0,
            "use_physics_loss": True,
            "physics_loss_weight": 0.1
        }
        
        self.output = {
            "output_dir": "./outputs",
            "experiment_name": "plancritic_training",
            "save_best_only": True
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
            
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        # Update configuration sections
        for section, values in config_data.items():
            if hasattr(self, section):
                getattr(self, section).update(values)
                
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_data = {
            "model": self.model,
            "training": self.training,
            "data": self.data,
            "physics": self.physics,
            "output": self.output
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)


class Trainer:
    """Trainer for trajectory critic model."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path(config.output["output_dir"]) / config.output["experiment_name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.physics_checker = None
        self.evaluator = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
    def setup_model(self) -> None:
        """Setup model and training components."""
        self.logger.info("Setting up model...")
        
        # Initialize feature encoders
        state_encoder = StateEncoder(
            input_dim=8,  # ego state dimension
            hidden_dim=64,
            output_dim=32
        )
        
        lane_encoder = LaneGraphEncoder(
            node_dim=6,  # lane features
            hidden_dim=64,
            output_dim=64
        )
        
        trajectory_encoder = TrajectoryEncoder(
            waypoint_dim=4,  # [x, y, vx, vy]
            seq_len=80,
            hidden_dim=64,
            output_dim=64
        )
        
        # Initialize base critic model
        base_critic = TrajectoryCritic(
            state_dim=32,
            lane_dim=64,
            cand_dim=64,
            hidden=self.config.model.get("hidden", 128),
            dropout=self.config.model["dropout"]
        )
        
        # Initialize model
        if self.config.data["max_candidates"] > 1:
            self.model = MultiCandidateCritic(base_critic)
        else:
            self.model = base_critic
        
        # Store encoders for use in training
        self.state_encoder = state_encoder
        self.lane_encoder = lane_encoder
        self.trajectory_encoder = trajectory_encoder
            
        self.model.to(self.device)
        self.state_encoder.to(self.device)
        self.lane_encoder.to(self.device)
        self.trajectory_encoder.to(self.device)
        
        # Initialize optimizer with all parameters
        all_params = list(self.model.parameters()) + \
                    list(self.state_encoder.parameters()) + \
                    list(self.lane_encoder.parameters()) + \
                    list(self.trajectory_encoder.parameters())
        
        self.optimizer = optim.AdamW(
            all_params,
            lr=self.config.training["learning_rate"],
            weight_decay=self.config.training["weight_decay"]
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training["num_epochs"]
        )
        
        # Initialize loss function
        self.criterion = CriticLoss()

        # Initialize physics loss (differentiable regularizer)
        self.physics_loss_fn = PhysicsLoss(
            ttc_threshold=self.config.physics.get("ttc_threshold", 3.0),
            jerk_threshold=self.config.physics.get("jerk_threshold", 2.0),
        )

        # Initialize physics checker with proper config
        from plancritic.eval.physics_checks import PhysicsConfig
        physics_config = PhysicsConfig(
            ttc_threshold=self.config.physics.get("ttc_threshold", 3.0),
            jerk_threshold=self.config.physics.get("jerk_threshold", 2.0),
            vehicle_length=self.config.physics.get("vehicle_length", 4.5),
            vehicle_width=self.config.physics.get("vehicle_width", 2.0),
            safety_margin=self.config.physics.get("safety_margin", 0.5)
        )
        self.physics_checker = PhysicsChecker(physics_config)
        
        # Initialize evaluator
        self.evaluator = CriticEvaluator()
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.model.parameters()) + \
                      sum(p.numel() for p in self.state_encoder.parameters()) + \
                      sum(p.numel() for p in self.lane_encoder.parameters()) + \
                      sum(p.numel() for p in self.trajectory_encoder.parameters())
        
        self.logger.info(f"Model initialized with {total_params} parameters")
        
    def setup_data(self) -> tuple:
        """Setup data loaders."""
        self.logger.info("Setting up data loaders...")

        dataset = self.config.data["dataset"]

        if dataset == "synthetic":
            seed = self.config.data.get("synthetic_seed", 42)
            n_cand = self.config.data["max_candidates"]
            horizon = self.config.data.get("prediction_horizon", 80)
            train_scenes = SyntheticScenarioGenerator(
                seed=seed, split="train",
                num_scenarios=self.config.data.get("synthetic_num_train", 400),
                num_candidates=n_cand, horizon=horizon,
            ).generate(with_labels=True)
            val_scenes = SyntheticScenarioGenerator(
                seed=seed, split="val",
                num_scenarios=self.config.data.get("synthetic_num_val", 80),
                num_candidates=n_cand, horizon=horizon,
            ).generate(with_labels=True)
            self.logger.info(
                f"Synthetic data: {len(train_scenes)} train, "
                f"{len(val_scenes)} val scenes (seed={seed})"
            )
        elif dataset in ("womd", "argoverse"):
            if dataset == "womd":
                from plancritic.data.adapters import WOMDAdapter, WOMDConfig
                adapter = WOMDAdapter(WOMDConfig(
                    data_dir=self.config.data["data_path"],
                    split="training",
                    max_scenes=self.config.data.get("max_scenes", None),
                ))
            else:
                from plancritic.data.adapters import ArgoverseAdapter, ArgoverseConfig
                adapter = ArgoverseAdapter(ArgoverseConfig(
                    data_dir=self.config.data["data_path"],
                    split="train",
                    max_scenes=self.config.data.get("max_scenes", None),
                ))
            scenes = list(adapter.load_scenes())
            split_idx = int(0.8 * len(scenes))
            train_scenes = scenes[:split_idx] if split_idx > 0 else scenes
            val_scenes = (scenes[split_idx:]
                          if split_idx > 0 and split_idx < len(scenes)
                          else scenes[:1])
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        collator = DataCollator(
            max_candidates=self.config.data["max_candidates"],
            trajectory_length=self.config.data.get("prediction_horizon", 80),
        )

        train_loader = DataLoader(
            train_scenes,
            batch_size=self.config.training["batch_size"],
            shuffle=True,
            collate_fn=collator.collate,
        )
        val_loader = DataLoader(
            val_scenes,
            batch_size=self.config.training["batch_size"],
            shuffle=False,
            collate_fn=collator.collate,
        )

        self.logger.info(f"Train loader: {len(train_loader)} batches")
        self.logger.info(f"Val loader: {len(val_loader)} batches")

        return train_loader, val_loader
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.state_encoder.train()
        self.lane_encoder.train()
        self.trajectory_encoder.train()
        
        total_loss = 0.0
        total_physics_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Encode features
            state_feats = self.state_encoder(batch["ego_states"])
            
            # For lane features, we'll use a simple placeholder since we don't have lane data yet
            batch_size = batch["ego_states"].shape[0]
            lane_feats = torch.zeros(batch_size, 64).to(self.device)
            
            # Encode trajectory candidates
            # trajectories shape: [B, num_candidates, seq_len, waypoint_dim]
            batch_size, num_candidates, seq_len, waypoint_dim = batch["trajectories"].shape
            
            # Reshape to process all candidates at once
            trajectories_flat = batch["trajectories"].view(batch_size * num_candidates, seq_len, waypoint_dim)
            cand_feats_flat = self.trajectory_encoder(trajectories_flat)
            
            # Reshape back to [B, num_candidates, output_dim]
            cand_feats = cand_feats_flat.view(batch_size, num_candidates, -1)
            
            outputs = self.model(
                state_feats=state_feats,
                lane_feats=lane_feats,
                cand_feats=cand_feats
            )
            
            # Use precomputed labels (synthetic) or compute on the fly
            if "physics_labels" in batch:
                physics_labels = batch["physics_labels"]
            else:
                physics_labels = self._generate_physics_labels(batch)

            # Compute loss
            loss, loss_components = self.criterion(outputs, physics_labels)

            total_loss_value = loss

            # Physics loss: differentiable TTC, jerk, and progress regularizer
            if self.config.physics["use_physics_loss"]:
                B, N, T, D = batch["trajectories"].shape

                # Flatten candidates into batch dimension
                traj_flat = batch["trajectories"].view(B * N, T, D)

                # Agent states (first 4 dims: x, y, vx, vy)
                agent_st = batch["agent_states"][:, :, :4]
                num_agents = agent_st.shape[1]
                agent_st_flat = agent_st.unsqueeze(1).expand(
                    -1, N, -1, -1
                ).reshape(B * N, num_agents, 4)

                agent_m = batch["agent_masks"]
                agent_m_flat = agent_m.unsqueeze(1).expand(
                    -1, N, -1
                ).reshape(B * N, num_agents)

                # Route waypoints (fall back to straight-ahead from ego)
                route_wp = batch.get("route_waypoints")
                if route_wp is None:
                    route_wp = torch.zeros(B, 10, 2, device=self.device)
                    for ri in range(10):
                        route_wp[:, ri, 0] = batch["ego_states"][:, 0] + (ri + 1) * 5.0
                        route_wp[:, ri, 1] = batch["ego_states"][:, 1]
                n_route = route_wp.shape[1]
                route_wp_flat = route_wp.unsqueeze(1).expand(
                    -1, N, -1, -1
                ).reshape(B * N, n_route, 2)

                # Flatten model predictions to match
                pred_flat = {
                    k: v.view(B * N, 1)
                    for k, v in outputs.items()
                    if k in ("risk", "comfort", "progress")
                }

                physics_loss, _ = self.physics_loss_fn(
                    pred_flat, traj_flat, agent_st_flat,
                    route_wp_flat, agent_m_flat
                )
                physics_loss_weighted = (
                    self.config.physics["physics_loss_weight"] * physics_loss
                )
                total_loss_value = total_loss_value + physics_loss_weighted
                total_physics_loss += physics_loss.item()
            
            # Backward pass
            total_loss_value.backward()
            
            # Gradient clipping
            if self.config.training["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training["gradient_clip"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_value.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss_value.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_physics_loss = total_physics_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "train_physics_loss": avg_physics_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.state_encoder.eval()
        self.lane_encoder.eval()
        self.trajectory_encoder.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Encode features
                state_feats = self.state_encoder(batch["ego_states"])
                
                # For lane features, we'll use a simple placeholder since we don't have lane data yet
                batch_size = batch["ego_states"].shape[0]
                lane_feats = torch.zeros(batch_size, 64).to(self.device)
                
                # Encode trajectory candidates
                # trajectories shape: [B, num_candidates, seq_len, waypoint_dim]
                batch_size, num_candidates, seq_len, waypoint_dim = batch["trajectories"].shape
                
                # Reshape to process all candidates at once
                trajectories_flat = batch["trajectories"].view(batch_size * num_candidates, seq_len, waypoint_dim)
                cand_feats_flat = self.trajectory_encoder(trajectories_flat)
                
                # Reshape back to [B, num_candidates, output_dim]
                cand_feats = cand_feats_flat.view(batch_size, num_candidates, -1)
                
                # Forward pass
                outputs = self.model(
                    state_feats=state_feats,
                    lane_feats=lane_feats,
                    cand_feats=cand_feats
                )
                
                # Use precomputed labels (synthetic) or compute on the fly
                if "physics_labels" in batch:
                    physics_labels = batch["physics_labels"]
                else:
                    physics_labels = self._generate_physics_labels(batch)

                # Compute loss
                loss, loss_components = self.criterion(outputs, physics_labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels for metrics
                all_predictions.append(outputs)
                all_labels.append(physics_labels)
                
        # Compute average loss
        avg_loss = total_loss / num_batches
        
        # Compute evaluation metrics
        # Convert tensor outputs to list of dicts for evaluator
        predictions_list = []
        labels_list = []
        
        for pred_batch, label_batch in zip(all_predictions, all_labels):
            batch_size = pred_batch["risk"].shape[0]
            num_candidates = pred_batch["risk"].shape[1]
            
            for i in range(batch_size):
                for j in range(num_candidates):
                    pred_dict = {
                        "risk": pred_batch["risk"][i, j, 0].item(),
                        "comfort": pred_batch["comfort"][i, j, 0].item(),
                        "progress": pred_batch["progress"][i, j, 0].item(),
                        "composite": pred_batch.get("score", pred_batch["risk"])[i, j, 0].item()
                    }
                    label_dict = {
                        "risk": label_batch["risk"][i, j, 0].item(),
                        "comfort": label_batch["comfort"][i, j, 0].item(),
                        "progress": label_batch["progress"][i, j, 0].item(),
                        "composite": label_batch.get("score", label_batch["risk"])[i, j, 0].item()
                    }
                    predictions_list.append(pred_dict)
                    labels_list.append(label_dict)
        
        metrics = self.evaluator.evaluate_critic(predictions_list, labels_list)
        
        return {
            "val_loss": avg_loss,
            **metrics.to_dict()
        }
        
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move a batch dict (possibly with nested dicts) to self.device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = {
                    dk: dv.to(self.device) if isinstance(dv, torch.Tensor) else dv
                    for dk, dv in v.items()
                }
            else:
                out[k] = v
        return out

    def _generate_physics_labels(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate physics-based pseudo-labels using PhysicsChecker.

        Computes collision risk (TTC + distance), comfort (jerk), and
        progress scores for each candidate trajectory in the batch.
        """
        trajectories = batch["trajectories"]  # [B, N, T, 4]
        batch_size, num_candidates, seq_len, _ = trajectories.shape
        device = trajectories.device

        risk_labels = torch.zeros(batch_size, num_candidates, 1, device=device)
        comfort_labels = torch.ones(batch_size, num_candidates, 1, device=device)
        progress_labels = torch.zeros(batch_size, num_candidates, 1, device=device)

        # Move to numpy for PhysicsChecker
        traj_np = trajectories.detach().cpu().numpy()
        agent_states_np = batch["agent_states"].detach().cpu().numpy()
        agent_masks_np = batch["agent_masks"].detach().cpu().numpy().astype(bool)
        ego_states_np = batch["ego_states"].detach().cpu().numpy()
        traj_masks = batch.get("trajectory_masks")

        route_wp_t = batch.get("route_waypoints")
        route_mask_t = batch.get("route_masks")

        dt = self.physics_checker.config.dt

        for b in range(batch_size):
            # Unpad route waypoints
            if route_wp_t is not None and route_mask_t is not None:
                rm = route_mask_t[b].detach().cpu().numpy().astype(bool)
                rw = route_wp_t[b].detach().cpu().numpy()[rm]
            else:
                rw = np.zeros((0, 2))

            scene = SceneData(
                ego_state=ego_states_np[b],
                lane_graph={},
                agent_states=agent_states_np[b],
                agent_mask=agent_masks_np[b],
                route_waypoints=rw,
                candidates=[],
                scene_id="",
                timestamp=0.0,
            )

            for c in range(num_candidates):
                if traj_masks is not None and not traj_masks[b, c]:
                    continue

                candidate = TrajectoryCandidate(
                    waypoints=traj_np[b, c],
                    timestamps=np.arange(seq_len) * dt,
                    metadata={},
                )

                risk_labels[b, c, 0] = self.physics_checker.compute_collision_risk(
                    candidate, scene
                )
                comfort_labels[b, c, 0] = self.physics_checker.compute_comfort_score(
                    candidate
                )
                progress_labels[b, c, 0] = self.physics_checker.compute_progress_score(
                    candidate, scene
                )

        return {
            "risk": risk_labels,
            "comfort": comfort_labels,
            "progress": progress_labels,
        }
        
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "training_history": self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
            
        self.logger.info(f"Saved checkpoint at epoch {epoch}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", [])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Setup model and data
        self.setup_model()
        train_loader, val_loader = self.setup_data()
        
        # Save initial config
        config_path = self.output_dir / "config.json"
        self.config.save_to_file(str(config_path))
        
        # Early stopping config
        patience = self.config.training.get("early_stopping_patience", 0)
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(self.current_epoch, self.config.training["num_epochs"]):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate epoch
            if epoch % self.config.training["eval_interval"] == 0:
                val_metrics = self.validate_epoch(val_loader)

                # Update learning rate
                self.scheduler.step()

                # Log metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                self.training_history.append(epoch_metrics)

                self.logger.info(f"Epoch {epoch}: " +
                               " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))

                # Save checkpoint
                is_best = val_metrics["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val_loss"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epoch % self.config.training["save_interval"] == 0 or is_best:
                    if not self.config.output["save_best_only"] or is_best:
                        self.save_checkpoint(epoch, is_best)

                # Early stopping
                if patience > 0 and epochs_without_improvement >= patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        self.logger.info("Training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PlanCritic trajectory critic")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="./data",
        help="Path to dataset"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="plancritic_training",
        help="Experiment name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "womd", "argoverse"],
        default="synthetic",
        help="Dataset to use"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        help="Path to checkpoint to resume from"
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
        # Initialize configuration
        config = TrainingConfig(args.config)
        
        # Override config with command line arguments
        if args.data_path:
            config.data["data_path"] = args.data_path
        if args.output_dir:
            config.output["output_dir"] = args.output_dir
        if args.experiment_name:
            config.output["experiment_name"] = args.experiment_name
        if args.dataset:
            config.data["dataset"] = args.dataset
        if args.batch_size:
            config.training["batch_size"] = args.batch_size
        if args.learning_rate:
            config.training["learning_rate"] = args.learning_rate
        if args.num_epochs:
            config.training["num_epochs"] = args.num_epochs
            
        # Initialize trainer
        trainer = Trainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
            
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()