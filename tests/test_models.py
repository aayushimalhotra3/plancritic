"""
Unit tests for PlanCritic neural network models.

This module tests the core neural network components including:
- TrajectoryCritic model
- Feature encoders (state, lane, trajectory)
- Loss functions
- Model initialization and forward passes
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from plancritic.models.critic import TrajectoryCritic, MultiCandidateCritic
from plancritic.models.encoders import (
    StateEncoder, LaneGraphEncoder, TrajectoryEncoder, 
    AttentionEncoder, PositionalEncoding
)
from plancritic.models.losses import (
    PhysicsInformedLoss, RankingLoss, CombinedLoss,
    collision_loss, comfort_loss, progress_loss
)


class TestTrajectoryCritic:
    """Test cases for the main TrajectoryCritic model."""
    
    @pytest.fixture
    def critic_config(self):
        """Standard configuration for trajectory critic."""
        return {
            'state_dim': 32,
            'lane_dim': 64,
            'cand_dim': 64,
            'hidden_dim': 128,
            'dropout': 0.1,
            'num_heads': 8,
            'num_layers': 3
        }
    
    @pytest.fixture
    def critic_model(self, critic_config):
        """Initialize trajectory critic model."""
        return TrajectoryCritic(**critic_config)
    
    def test_model_initialization(self, critic_model, critic_config):
        """Test that model initializes with correct architecture."""
        assert isinstance(critic_model, TrajectoryCritic)
        assert critic_model.state_dim == critic_config['state_dim']
        assert critic_model.lane_dim == critic_config['lane_dim']
        assert critic_model.cand_dim == critic_config['cand_dim']
        
        # Check that all components are initialized
        assert hasattr(critic_model, 'state_encoder')
        assert hasattr(critic_model, 'lane_encoder')
        assert hasattr(critic_model, 'trajectory_encoder')
        assert hasattr(critic_model, 'scoring_head')
    
    def test_forward_pass_shapes(self, critic_model):
        """Test forward pass with correct input/output shapes."""
        batch_size = 4
        seq_len = 80
        num_agents = 8
        num_lanes = 20
        
        # Create mock inputs
        agent_states = torch.randn(batch_size, num_agents, seq_len, 6)  # [x, y, vx, vy, ax, ay]
        lane_graph = torch.randn(batch_size, num_lanes, 64)  # lane features
        trajectory_candidates = torch.randn(batch_size, 3, seq_len, 4)  # 3 candidates
        
        # Forward pass
        scores = critic_model(agent_states, lane_graph, trajectory_candidates)
        
        # Check output shape
        assert scores.shape == (batch_size, 3)  # batch_size x num_candidates
        assert torch.all(scores >= 0) and torch.all(scores <= 1)  # scores in [0, 1]
    
    def test_single_trajectory_scoring(self, critic_model):
        """Test scoring a single trajectory."""
        batch_size = 2
        seq_len = 50
        
        agent_states = torch.randn(batch_size, 5, seq_len, 6)
        lane_graph = torch.randn(batch_size, 10, 64)
        single_trajectory = torch.randn(batch_size, seq_len, 4)
        
        score = critic_model.score_trajectory(agent_states, lane_graph, single_trajectory)
        
        assert score.shape == (batch_size,)
        assert torch.all(score >= 0) and torch.all(score <= 1)
    
    def test_gradient_flow(self, critic_model):
        """Test that gradients flow properly through the model."""
        batch_size = 2
        seq_len = 40
        
        agent_states = torch.randn(batch_size, 3, seq_len, 6, requires_grad=True)
        lane_graph = torch.randn(batch_size, 5, 64, requires_grad=True)
        trajectory_candidates = torch.randn(batch_size, 2, seq_len, 4, requires_grad=True)
        
        scores = critic_model(agent_states, lane_graph, trajectory_candidates)
        loss = scores.sum()
        loss.backward()
        
        # Check that gradients exist
        assert agent_states.grad is not None
        assert lane_graph.grad is not None
        assert trajectory_candidates.grad is not None
        
        # Check that model parameters have gradients
        for param in critic_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_device_compatibility(self, critic_config):
        """Test model works on different devices."""
        model = TrajectoryCritic(**critic_config)
        
        # Test CPU
        inputs = self._create_sample_inputs(device='cpu')
        scores_cpu = model(*inputs)
        assert scores_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            inputs_gpu = self._create_sample_inputs(device='cuda')
            scores_gpu = model_gpu(*inputs_gpu)
            assert scores_gpu.device.type == 'cuda'
    
    def _create_sample_inputs(self, device='cpu'):
        """Helper to create sample inputs for testing."""
        agent_states = torch.randn(2, 4, 30, 6, device=device)
        lane_graph = torch.randn(2, 8, 64, device=device)
        trajectory_candidates = torch.randn(2, 3, 30, 4, device=device)
        return agent_states, lane_graph, trajectory_candidates


class TestMultiCandidateCritic:
    """Test cases for multi-candidate trajectory evaluation."""
    
    @pytest.fixture
    def multi_critic(self):
        """Initialize multi-candidate critic."""
        return MultiCandidateCritic(
            state_dim=32,
            lane_dim=64,
            cand_dim=64,
            max_candidates=5,
            use_attention=True
        )
    
    def test_variable_candidate_numbers(self, multi_critic):
        """Test handling variable numbers of candidates."""
        batch_size = 3
        seq_len = 60
        
        agent_states = torch.randn(batch_size, 6, seq_len, 6)
        lane_graph = torch.randn(batch_size, 12, 64)
        
        # Test with different numbers of candidates
        for num_candidates in [1, 3, 5]:
            candidates = torch.randn(batch_size, num_candidates, seq_len, 4)
            scores = multi_critic(agent_states, lane_graph, candidates)
            assert scores.shape == (batch_size, num_candidates)
    
    def test_candidate_ranking(self, multi_critic):
        """Test that model can rank candidates appropriately."""
        batch_size = 2
        seq_len = 40
        num_candidates = 4
        
        agent_states = torch.randn(batch_size, 4, seq_len, 6)
        lane_graph = torch.randn(batch_size, 8, 64)
        candidates = torch.randn(batch_size, num_candidates, seq_len, 4)
        
        scores = multi_critic(agent_states, lane_graph, candidates)
        rankings = torch.argsort(scores, dim=1, descending=True)
        
        assert rankings.shape == (batch_size, num_candidates)
        # Check that rankings contain all indices
        for b in range(batch_size):
            assert set(rankings[b].tolist()) == set(range(num_candidates))


class TestEncoders:
    """Test cases for feature encoders."""
    
    def test_state_encoder(self):
        """Test state encoder functionality."""
        encoder = StateEncoder(input_dim=6, hidden_dim=64, output_dim=32)
        
        batch_size, num_agents, seq_len = 3, 5, 50
        states = torch.randn(batch_size, num_agents, seq_len, 6)
        
        encoded = encoder(states)
        assert encoded.shape == (batch_size, num_agents, 32)
    
    def test_lane_graph_encoder(self):
        """Test lane graph encoder with GNN layers."""
        encoder = LaneGraphEncoder(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            output_dim=64,
            num_layers=2
        )
        
        batch_size, num_nodes = 2, 15
        node_features = torch.randn(batch_size, num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, 30))  # 30 edges
        edge_features = torch.randn(30, 32)
        
        encoded = encoder(node_features, edge_index, edge_features)
        assert encoded.shape == (batch_size, num_nodes, 64)
    
    def test_trajectory_encoder(self):
        """Test trajectory encoder with temporal modeling."""
        encoder = TrajectoryEncoder(
            input_dim=4,
            hidden_dim=128,
            output_dim=64,
            num_layers=3,
            use_attention=True
        )
        
        batch_size, seq_len = 4, 80
        trajectory = torch.randn(batch_size, seq_len, 4)
        
        encoded = encoder(trajectory)
        assert encoded.shape == (batch_size, 64)
    
    def test_attention_encoder(self):
        """Test attention-based encoder."""
        encoder = AttentionEncoder(
            input_dim=128,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )
        
        batch_size, seq_len = 2, 60
        inputs = torch.randn(batch_size, seq_len, 128)
        
        encoded = encoder(inputs)
        assert encoded.shape == (batch_size, seq_len, 128)
    
    def test_positional_encoding(self):
        """Test positional encoding for sequences."""
        pos_enc = PositionalEncoding(d_model=128, max_len=100)
        
        batch_size, seq_len = 3, 80
        inputs = torch.randn(batch_size, seq_len, 128)
        
        encoded = pos_enc(inputs)
        assert encoded.shape == inputs.shape
        
        # Test that encoding is deterministic
        encoded2 = pos_enc(inputs)
        assert torch.allclose(encoded, encoded2)


class TestLossFunctions:
    """Test cases for loss functions."""
    
    def test_physics_informed_loss(self):
        """Test physics-informed loss computation."""
        loss_fn = PhysicsInformedLoss(
            collision_weight=0.4,
            comfort_weight=0.3,
            progress_weight=0.3
        )
        
        batch_size = 4
        predictions = torch.randn(batch_size, 3)  # 3 candidates
        physics_scores = torch.randn(batch_size, 3)
        
        loss = loss_fn(predictions, physics_scores)
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_ranking_loss(self):
        """Test ranking loss for trajectory comparison."""
        loss_fn = RankingLoss(margin=0.1)
        
        batch_size = 3
        scores = torch.tensor([[0.8, 0.6, 0.4], [0.7, 0.5, 0.3], [0.9, 0.7, 0.5]])
        rankings = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])  # ground truth rankings
        
        loss = loss_fn(scores, rankings)
        assert loss.item() >= 0
    
    def test_combined_loss(self):
        """Test combined loss with multiple components."""
        loss_fn = CombinedLoss(
            physics_weight=0.5,
            ranking_weight=0.3,
            regularization_weight=0.2
        )
        
        batch_size = 2
        predictions = torch.randn(batch_size, 4, requires_grad=True)
        physics_scores = torch.randn(batch_size, 4)
        rankings = torch.randint(0, 4, (batch_size, 4))
        
        loss = loss_fn(predictions, physics_scores, rankings)
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_collision_loss(self):
        """Test collision-specific loss computation."""
        batch_size, seq_len = 3, 50
        trajectory = torch.randn(batch_size, seq_len, 4)
        other_agents = torch.randn(batch_size, 5, seq_len, 4)  # 5 other agents
        
        loss = collision_loss(trajectory, other_agents, threshold=2.0)
        assert loss.shape == (batch_size,)
        assert torch.all(loss >= 0)
    
    def test_comfort_loss(self):
        """Test comfort-based loss computation."""
        batch_size, seq_len = 2, 60
        trajectory = torch.randn(batch_size, seq_len, 4)
        
        loss = comfort_loss(trajectory, dt=0.1, max_accel=4.0, max_jerk=3.0)
        assert loss.shape == (batch_size,)
        assert torch.all(loss >= 0)
    
    def test_progress_loss(self):
        """Test progress-based loss computation."""
        batch_size, seq_len = 4, 40
        trajectory = torch.randn(batch_size, seq_len, 4)
        target_position = torch.randn(batch_size, 2)
        
        loss = progress_loss(trajectory, target_position)
        assert loss.shape == (batch_size,)
        assert torch.all(loss >= 0)


class TestModelIntegration:
    """Integration tests for complete model pipeline."""
    
    def test_end_to_end_training_step(self):
        """Test complete training step with loss computation."""
        # Initialize model and loss
        model = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64)
        loss_fn = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create sample batch
        batch_size = 2
        agent_states = torch.randn(batch_size, 4, 50, 6)
        lane_graph = torch.randn(batch_size, 10, 64)
        candidates = torch.randn(batch_size, 3, 50, 4)
        physics_scores = torch.randn(batch_size, 3)
        rankings = torch.randint(0, 3, (batch_size, 3))
        
        # Training step
        optimizer.zero_grad()
        predictions = model(agent_states, lane_graph, candidates)
        loss = loss_fn(predictions, physics_scores, rankings)
        loss.backward()
        optimizer.step()
        
        # Verify training step completed
        assert loss.item() >= 0
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_evaluation_mode(self):
        """Test model behavior in evaluation mode."""
        model = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64, dropout=0.5)
        
        # Create inputs
        inputs = self._create_sample_inputs()
        
        # Test training mode
        model.train()
        scores_train1 = model(*inputs)
        scores_train2 = model(*inputs)
        
        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            scores_eval1 = model(*inputs)
            scores_eval2 = model(*inputs)
        
        # In eval mode, outputs should be deterministic (no dropout)
        assert torch.allclose(scores_eval1, scores_eval2, atol=1e-6)
    
    def test_model_state_dict_save_load(self):
        """Test model serialization and loading."""
        model1 = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state dict
        model2 = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64)
        model2.load_state_dict(state_dict)
        
        # Test that models produce same outputs
        inputs = self._create_sample_inputs()
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            scores1 = model1(*inputs)
            scores2 = model2(*inputs)
        
        assert torch.allclose(scores1, scores2, atol=1e-6)
    
    def _create_sample_inputs(self):
        """Helper to create sample inputs."""
        agent_states = torch.randn(2, 3, 40, 6)
        lane_graph = torch.randn(2, 8, 64)
        candidates = torch.randn(2, 2, 40, 4)
        return agent_states, lane_graph, candidates


class TestModelRobustness:
    """Test model robustness and edge cases."""
    
    def test_empty_inputs(self):
        """Test model behavior with minimal inputs."""
        model = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64)
        
        # Test with minimal sequence length
        agent_states = torch.randn(1, 1, 1, 6)
        lane_graph = torch.randn(1, 1, 64)
        candidates = torch.randn(1, 1, 1, 4)
        
        scores = model(agent_states, lane_graph, candidates)
        assert scores.shape == (1, 1)
    
    def test_large_inputs(self):
        """Test model with large input dimensions."""
        model = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64)
        
        # Test with large inputs
        agent_states = torch.randn(1, 50, 200, 6)  # many agents, long sequence
        lane_graph = torch.randn(1, 100, 64)  # many lanes
        candidates = torch.randn(1, 10, 200, 4)  # many candidates
        
        scores = model(agent_states, lane_graph, candidates)
        assert scores.shape == (1, 10)
    
    def test_numerical_stability(self):
        """Test model numerical stability with extreme values."""
        model = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64)
        
        # Test with very small values
        small_inputs = [torch.randn(1, 2, 10, d) * 1e-6 for d in [6, 64, 4]]
        small_inputs[1] = torch.randn(1, 5, 64) * 1e-6
        scores_small = model(*small_inputs)
        assert torch.all(torch.isfinite(scores_small))
        
        # Test with large values
        large_inputs = [torch.randn(1, 2, 10, d) * 1e3 for d in [6, 64, 4]]
        large_inputs[1] = torch.randn(1, 5, 64) * 1e3
        scores_large = model(*large_inputs)
        assert torch.all(torch.isfinite(scores_large))


if __name__ == "__main__":
    pytest.main([__file__])