#!/usr/bin/env python3
"""
Basic PlanCritic Demo

This script demonstrates the core functionality of PlanCritic:
- Creating a trajectory critic model
- Generating synthetic trajectory data
- Evaluating trajectory quality
- Physics-based analysis
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

# Import PlanCritic components
from plancritic.models.critic import TrajectoryCritic
from plancritic.models.encoders import StateEncoder, TrajectoryEncoder
from plancritic.eval.physics_checks import PhysicsChecker
from plancritic.eval.metrics import CriticEvaluator


def create_synthetic_trajectory_data(
    batch_size: int = 4,
    sequence_length: int = 50,
    num_agents: int = 8,
    state_dim: int = 7  # [x, y, vx, vy, heading, length, width]
) -> Dict[str, torch.Tensor]:
    """
    Create synthetic trajectory data for demonstration.
    
    Args:
        batch_size: Number of scenarios
        sequence_length: Number of time steps
        num_agents: Number of agents per scenario
        state_dim: Dimension of agent state
    
    Returns:
        Dictionary containing synthetic trajectory data
    """
    print(f"Creating synthetic data: {batch_size} scenarios, {sequence_length} steps, {num_agents} agents")
    
    # Generate random trajectories
    trajectories = torch.randn(batch_size, sequence_length, num_agents, state_dim)
    
    # Make trajectories more realistic by ensuring smooth motion
    for b in range(batch_size):
        for a in range(num_agents):
            # Start with random initial position and velocity
            x0, y0 = torch.randn(2) * 10  # Initial position
            vx0, vy0 = torch.randn(2) * 5  # Initial velocity
            
            # Generate smooth trajectory
            for t in range(sequence_length):
                dt = 0.1  # Time step
                
                # Add some noise to velocity
                vx_noise = torch.randn(1) * 0.5
                vy_noise = torch.randn(1) * 0.5
                
                if t == 0:
                    trajectories[b, t, a, 0] = x0  # x position
                    trajectories[b, t, a, 1] = y0  # y position
                    trajectories[b, t, a, 2] = vx0 + vx_noise  # vx
                    trajectories[b, t, a, 3] = vy0 + vy_noise  # vy
                else:
                    # Update position based on velocity
                    trajectories[b, t, a, 0] = trajectories[b, t-1, a, 0] + trajectories[b, t-1, a, 2] * dt
                    trajectories[b, t, a, 1] = trajectories[b, t-1, a, 1] + trajectories[b, t-1, a, 3] * dt
                    trajectories[b, t, a, 2] = trajectories[b, t-1, a, 2] + vx_noise
                    trajectories[b, t, a, 3] = trajectories[b, t-1, a, 3] + vy_noise
                
                # Set heading based on velocity
                trajectories[b, t, a, 4] = torch.atan2(trajectories[b, t, a, 3], trajectories[b, t, a, 2])
                
                # Set vehicle dimensions
                trajectories[b, t, a, 5] = 4.5  # length
                trajectories[b, t, a, 6] = 2.0  # width
    
    # Create agent masks (which agents are valid)
    agent_masks = torch.ones(batch_size, num_agents, dtype=torch.bool)
    
    # Create simple lane graph (dummy data)
    lane_positions = torch.randn(batch_size, 20, 2)  # 20 lane points per scenario
    lane_features = torch.randn(batch_size, 20, 4)   # Lane features
    
    return {
        'trajectories': trajectories,
        'agent_masks': agent_masks,
        'lane_positions': lane_positions,
        'lane_features': lane_features
    }


def demonstrate_trajectory_critic():
    """Demonstrate the TrajectoryCritic model."""
    print("\n=== PlanCritic Trajectory Evaluation Demo ===\n")
    
    # Model configuration
    config = {
        'state_dim': 32,
        'lane_dim': 64,
        'cand_dim': 64,
        'hidden': 128,
        'dropout': 0.1
    }
    
    print("1. Creating TrajectoryCritic model...")
    
    # Create the critic model
    critic = TrajectoryCritic(
        state_dim=config['state_dim'],
        lane_dim=config['lane_dim'],
        cand_dim=config['cand_dim'],
        hidden=config['hidden'],
        dropout=config['dropout']
    )
    
    print(f"   Model created with {sum(p.numel() for p in critic.parameters())} parameters")
    
    # Generate synthetic data
    print("\n2. Generating synthetic trajectory data...")
    data = create_synthetic_trajectory_data()
    
    # Create feature tensors that match the model's expected input
    batch_size = data['trajectories'].shape[0]
    
    # Create state features (ego vehicle state)
    state_feats = torch.randn(batch_size, config['state_dim'])
    
    # Create lane features 
    lane_feats = torch.randn(batch_size, config['lane_dim'])
    
    # Create candidate trajectory features
    cand_feats = torch.randn(batch_size, config['cand_dim'])
    
    # Evaluate trajectories
    print("\n3. Evaluating trajectory quality...")
    critic.eval()
    with torch.no_grad():
        # Forward pass through the critic
        scores = critic(
            state_feats=state_feats,
            lane_feats=lane_feats,
            cand_feats=cand_feats
        )
        
        print(f"   Trajectory scores:")
        for score_type, score_tensor in scores.items():
            print(f"     {score_type}: shape={score_tensor.shape}, mean={score_tensor.mean().item():.4f}")
    
    return critic, data, scores


def demonstrate_physics_analysis(data: Dict[str, torch.Tensor]):
    """Demonstrate physics-based trajectory analysis."""
    print("\n4. Physics-based trajectory analysis...")
    
    # Import required classes
    from plancritic.eval.physics_checks import PhysicsChecker, PhysicsConfig
    from plancritic.data.samplers import TrajectoryCandidate, SceneData
    
    # Create physics checker with proper config
    config = PhysicsConfig(
        ttc_threshold=3.0,
        jerk_threshold=2.0,
        vehicle_length=4.5,
        vehicle_width=2.0,
        safety_margin=0.5
    )
    physics_checker = PhysicsChecker(config)
    
    trajectories = data['trajectories']
    batch_size, seq_len, num_agents, _ = trajectories.shape
    
    print(f"   Analyzing {batch_size} scenarios with {num_agents} agents each...")
    
    # Analyze each scenario
    physics_scores = []
    
    for b in range(min(3, batch_size)):  # Analyze first 3 scenarios
        scenario_traj = trajectories[b]  # [seq_len, num_agents, state_dim]
        
        # Create mock trajectory candidate and scene data
        # Extract ego trajectory (first agent)
        ego_traj = scenario_traj[:, 0, :].numpy()  # [seq_len, state_dim]
        waypoints = ego_traj[:, :2]  # Extract x, y positions
        
        # Create trajectory candidate with proper format
        # waypoints should be [T, 4] format: (x, y, vx, vy)
        if ego_traj.shape[1] >= 4:
            waypoints_4d = ego_traj[:, :4]  # Use first 4 dimensions
        else:
            # Pad with zeros for velocity if not available
            waypoints_4d = np.zeros((len(waypoints), 4))
            waypoints_4d[:, :2] = waypoints  # x, y positions
        
        candidate = TrajectoryCandidate(
            waypoints=waypoints_4d,
            timestamps=np.arange(len(waypoints_4d)) * 0.1,
            metadata={"planner_id": f"scenario_{b}", "cost": 1.0, "feasible": True}
        )
        
        # Create scene data with other agents
        # Convert other agents to proper format for SceneData
        agent_states = []
        for agent_idx in range(1, min(3, num_agents)):  # Use up to 2 other agents
            agent_traj = scenario_traj[:, agent_idx, :].numpy()
            if len(agent_traj) > 0:
                # Use initial state of agent [8] format expected
                if agent_traj.shape[1] >= 8:
                    agent_state = agent_traj[0, :8]
                else:
                    # Pad with zeros if not enough dimensions
                    agent_state = np.zeros(8)
                    agent_state[:agent_traj.shape[1]] = agent_traj[0]
                agent_states.append(agent_state)
        
        # Convert to numpy array and create mask
        if agent_states:
            agent_states_array = np.array(agent_states)
            agent_mask = np.ones(len(agent_states), dtype=bool)
        else:
            agent_states_array = np.zeros((0, 8))
            agent_mask = np.zeros(0, dtype=bool)
        
        # Create simple lane graph (mock data)
        lane_graph = {
            'nodes': np.random.randn(10, 8),  # Mock lane nodes
            'edges': np.array([[i, i+1] for i in range(9)]),  # Sequential connections
            'edge_features': np.random.randn(9, 4)
        }
        
        scene = SceneData(
            ego_state=ego_traj[0] if len(ego_traj[0]) >= 8 else np.pad(ego_traj[0], (0, max(0, 8-len(ego_traj[0])))),
            lane_graph=lane_graph,
            agent_states=agent_states_array,
            agent_mask=agent_mask,
            route_waypoints=waypoints,  # Use trajectory as route
            candidates=[candidate],  # Include the candidate we created
            scene_id=f"scene_{b}",
            timestamp=0.0
        )
        
        try:
            # Evaluate trajectory
            scores = physics_checker.evaluate_trajectory(candidate, scene)
            physics_scores.append(scores)
            
            print(f"   Scenario {b+1} Physics Scores:")
            print(f"     Risk: {scores['risk']:.3f}")
            print(f"     Comfort: {scores['comfort']:.3f}")
            print(f"     Progress: {scores['progress']:.3f}")
            print(f"     Composite: {scores['composite']:.3f}")
            
        except Exception as e:
            print(f"   Scenario {b+1}: Analysis failed - {e}")
            physics_scores.append({
                'risk': 0.5, 'comfort': 0.5, 'progress': 0.5, 'composite': 0.5
            })
    
    if physics_scores:
        avg_scores = {
            key: np.mean([score[key] for score in physics_scores])
            for key in physics_scores[0].keys()
        }
        print(f"\n   Average Physics Scores:")
        for key, value in avg_scores.items():
            print(f"     {key.capitalize()}: {value:.3f}")
    
    return physics_scores


def demonstrate_evaluation_metrics(scores: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
    """Demonstrate evaluation metrics computation."""
    print("\n5. Computing evaluation metrics...")
    
    from plancritic.eval.metrics import CriticEvaluator
    
    # Create evaluator
    evaluator = CriticEvaluator()
    
    # Extract score tensor from the scores dictionary
    score_tensor = scores['score']  # This should be the main score tensor
    batch_size = score_tensor.shape[0]
    
    # Generate synthetic ground truth scores for demonstration
    gt_scores = torch.randn_like(score_tensor)
    gt_scores = torch.sigmoid(gt_scores)  # Normalize to [0, 1]
    
    print(f"   Evaluating {batch_size} trajectory predictions...")
    
    # Compute metrics
    try:
        # Convert to numpy for evaluation
        pred_scores = score_tensor.detach().cpu().numpy()
        true_scores = gt_scores.detach().cpu().numpy()
        
        # Compute correlation
        from scipy.stats import pearsonr, spearmanr
        
        # Flatten arrays for correlation computation
        pred_flat = pred_scores.flatten()
        true_flat = true_scores.flatten()
        
        pearson_corr, pearson_p = pearsonr(pred_flat, true_flat)
        spearman_corr, spearman_p = spearmanr(pred_flat, true_flat)
        
        # Compute MSE and MAE
        mse = np.mean((pred_flat - true_flat) ** 2)
        mae = np.mean(np.abs(pred_flat - true_flat))
        
        print(f"   Evaluation Metrics:")
        print(f"     Pearson Correlation: {pearson_corr:.3f} (p={pearson_p:.3f})")
        print(f"     Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
        print(f"     Mean Squared Error: {mse:.3f}")
        print(f"     Mean Absolute Error: {mae:.3f}")
        
        # Compute ranking metrics if we have multiple candidates
        if batch_size > 1:
            # Compute ranking accuracy (top-1)
            pred_ranks = np.argsort(-pred_flat)
            true_ranks = np.argsort(-true_flat)
            
            top1_acc = (pred_ranks[0] == true_ranks[0])
            print(f"     Top-1 Ranking Accuracy: {top1_acc}")
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'mse': mse,
            'mae': mae
        }
        
    except Exception as e:
        print(f"   Metrics computation failed: {e}")
        return {
            'pearson_correlation': 0.0,
            'spearman_correlation': 0.0,
            'mse': 1.0,
            'mae': 1.0
        }


def main():
    """Main demonstration function."""
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run the main demonstration
        critic, data, scores = demonstrate_trajectory_critic()
        
        # Physics analysis
        physics_scores = demonstrate_physics_analysis(data)
        
        # Evaluation metrics
        metrics = demonstrate_evaluation_metrics(scores, data)
        
        print("\n=== Demo completed successfully! ===")
        print("\nThis demo showed:")
        print("✓ Creating and using a TrajectoryCritic model")
        print("✓ Generating synthetic trajectory data")
        print("✓ Evaluating trajectory quality with neural networks")
        print("✓ Physics-based trajectory analysis")
        print("✓ Computing evaluation metrics")
        
        print("\nNext steps:")
        print("- Try with real trajectory data (WOMD, Argoverse)")
        print("- Train the model on your dataset")
        print("- Use the CLI tools for batch processing")
        print("- Explore the web visualization interface")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("This might be due to missing dependencies or data.")
        print("Please check the installation and try again.")
        raise


if __name__ == "__main__":
    main()