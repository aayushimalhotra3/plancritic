#!/usr/bin/env python3
"""
Full training run on synthetic benchmark with evaluation and plots.

Trains the critic, saves checkpoint, evaluates on held-out test split,
and generates AUROC / Spearman / calibration plots to results/.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plancritic.cli.train import TrainingConfig, Trainer
from plancritic.data.synthetic import SyntheticScenarioGenerator
from plancritic.data.samplers import DataCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
PLOTS_DIR = RESULTS_DIR / "plots"

SEED = 42


# ── helpers ──────────────────────────────────────────────────────────

def collect_test_predictions(trainer, test_scenes):
    """Run the trained model on test scenes, return predictions + labels."""
    collator = DataCollator(
        max_candidates=trainer.config.data["max_candidates"],
        trajectory_length=trainer.config.data.get("prediction_horizon", 80),
    )

    trainer.model.eval()
    trainer.state_encoder.eval()
    trainer.trajectory_encoder.eval()

    all_pred_risk, all_pred_comfort, all_pred_progress = [], [], []
    all_gt_risk, all_gt_comfort, all_gt_progress = [], [], []
    all_gt_collided = []

    # process in batches
    bs = trainer.config.training["batch_size"]
    with torch.no_grad():
        for start in range(0, len(test_scenes), bs):
            scenes_batch = test_scenes[start : start + bs]
            batch = collator.collate(scenes_batch)
            batch = trainer._batch_to_device(batch)

            state_feats = trainer.state_encoder(batch["ego_states"])
            batch_size = batch["ego_states"].shape[0]
            lane_feats = torch.zeros(batch_size, 64, device=trainer.device)

            B, N, T, D = batch["trajectories"].shape
            traj_flat = batch["trajectories"].view(B * N, T, D)
            cand_feats = trainer.trajectory_encoder(traj_flat).view(B, N, -1)

            outputs = trainer.model(
                state_feats=state_feats,
                lane_feats=lane_feats,
                cand_feats=cand_feats,
            )

            labels = batch["physics_labels"]
            mask = batch["trajectory_masks"]  # [B, N]

            for i in range(B):
                for j in range(N):
                    if not mask[i, j]:
                        continue
                    all_pred_risk.append(outputs["risk"][i, j, 0].item())
                    all_pred_comfort.append(outputs["comfort"][i, j, 0].item())
                    all_pred_progress.append(outputs["progress"][i, j, 0].item())
                    all_gt_risk.append(labels["risk"][i, j, 0].item())
                    all_gt_comfort.append(labels["comfort"][i, j, 0].item())
                    all_gt_progress.append(labels["progress"][i, j, 0].item())
                    all_gt_collided.append(labels["collided"][i, j, 0].item())

    return {
        "pred_risk": np.array(all_pred_risk),
        "pred_comfort": np.array(all_pred_comfort),
        "pred_progress": np.array(all_pred_progress),
        "gt_risk": np.array(all_gt_risk),
        "gt_comfort": np.array(all_gt_comfort),
        "gt_progress": np.array(all_gt_progress),
        "gt_collided": np.array(all_gt_collided),
    }


def compute_metrics(data):
    """Compute AUROC, Spearman, MAE for all heads."""
    metrics = {}

    # AUROC: predicted risk vs binary collided
    collided = data["gt_collided"].astype(int)
    if len(np.unique(collided)) == 2:
        metrics["collision_auroc"] = roc_auc_score(collided, data["pred_risk"])
    else:
        metrics["collision_auroc"] = float("nan")

    # Spearman correlations (pred vs continuous physics label)
    for head in ("risk", "comfort", "progress"):
        rho, p = spearmanr(data[f"pred_{head}"], data[f"gt_{head}"])
        metrics[f"spearman_{head}"] = rho if not np.isnan(rho) else 0.0

    # MAE
    for head in ("risk", "comfort", "progress"):
        metrics[f"mae_{head}"] = float(
            np.mean(np.abs(data[f"pred_{head}"] - data[f"gt_{head}"]))
        )

    return metrics


# ── plotting ─────────────────────────────────────────────────────────

def plot_auroc(data, save_path):
    collided = data["gt_collided"].astype(int)
    if len(np.unique(collided)) < 2:
        log.warning("Cannot plot AUROC: only one class present")
        return
    fpr, tpr, _ = roc_curve(collided, data["pred_risk"])
    auc_val = roc_auc_score(collided, data["pred_risk"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Collision Detection ROC")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved AUROC plot to {save_path}")


def plot_spearman(data, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, head in zip(axes, ("risk", "comfort", "progress")):
        pred = data[f"pred_{head}"]
        gt = data[f"gt_{head}"]
        rho, _ = spearmanr(pred, gt)
        ax.scatter(gt, pred, alpha=0.3, s=8)
        ax.plot([0, 1], [0, 1], ls="--", color="grey")
        ax.set_xlabel(f"Physics label ({head})")
        ax.set_ylabel(f"Predicted ({head})")
        ax.set_title(f"{head}  ρ={rho:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved Spearman scatter to {save_path}")


def plot_calibration(data, save_path, n_bins=10):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, head in zip(axes, ("risk", "comfort", "progress")):
        pred = data[f"pred_{head}"]
        gt = data[f"gt_{head}"]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means_pred = []
        bin_means_gt = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (pred >= lo) & (pred < hi)
            if mask.sum() > 0:
                bin_means_pred.append(pred[mask].mean())
                bin_means_gt.append(gt[mask].mean())

        ax.plot([0, 1], [0, 1], ls="--", color="grey")
        if bin_means_pred:
            ax.plot(bin_means_pred, bin_means_gt, "o-", markersize=5)
        ax.set_xlabel(f"Mean predicted ({head})")
        ax.set_ylabel(f"Mean physics label ({head})")
        ax.set_title(f"Calibration: {head}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved calibration plot to {save_path}")


def plot_training_loss(history, save_path):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved loss plot to {save_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── configure ──
    config = TrainingConfig()
    config.data["dataset"] = "synthetic"
    config.data["synthetic_seed"] = SEED
    config.data["synthetic_num_train"] = 1000
    config.data["synthetic_num_val"] = 200
    config.data["max_candidates"] = 8
    config.data["prediction_horizon"] = 80
    config.training["batch_size"] = 32
    config.training["num_epochs"] = 80
    config.training["early_stopping_patience"] = 10
    config.training["learning_rate"] = 3e-4
    config.training["eval_interval"] = 1
    config.training["save_interval"] = 10
    config.training["gradient_clip"] = 1.0
    config.physics["use_physics_loss"] = True
    config.physics["physics_loss_weight"] = 0.1
    config.output["output_dir"] = str(CHECKPOINT_DIR)
    config.output["experiment_name"] = "synthetic_v1"
    config.output["save_best_only"] = False

    # ── train ──
    log.info("Starting training run")
    trainer = Trainer(config)
    trainer.train()

    # save final checkpoint explicitly
    final_ckpt = CHECKPOINT_DIR / "synthetic_v1" / "final_model.pt"
    torch.save(
        {
            "epoch": config.training["num_epochs"],
            "model_state_dict": trainer.model.state_dict(),
            "state_encoder_state_dict": trainer.state_encoder.state_dict(),
            "trajectory_encoder_state_dict": trainer.trajectory_encoder.state_dict(),
            "config": config.__dict__,
        },
        final_ckpt,
    )
    log.info(f"Saved final checkpoint to {final_ckpt}")

    # ── evaluate on test split ──
    log.info("Generating test set")
    test_scenes = SyntheticScenarioGenerator(
        seed=SEED,
        split="test",
        num_scenarios=200,
        num_candidates=config.data["max_candidates"],
        horizon=config.data["prediction_horizon"],
    ).generate(with_labels=True)

    log.info(f"Evaluating on {len(test_scenes)} test scenes")
    data = collect_test_predictions(trainer, test_scenes)
    metrics = compute_metrics(data)

    # ── training history for loss plot ──
    history = []
    for i, h in enumerate(trainer.training_history):
        history.append({"epoch": i, **h})

    # ── plots ──
    plot_auroc(data, PLOTS_DIR / "auroc.png")
    plot_spearman(data, PLOTS_DIR / "spearman.png")
    plot_calibration(data, PLOTS_DIR / "calibration.png")
    if history:
        plot_training_loss(history, PLOTS_DIR / "training_loss.png")

    # ── save metrics ──
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {metrics_path}")

    # ── print summary ──
    print("\n" + "=" * 50)
    print("FINAL TEST METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:25s}  {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
