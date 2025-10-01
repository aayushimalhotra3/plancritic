"""
Pytest configuration and shared fixtures for PlanCritic tests.

This module provides common test fixtures, configuration, and utilities
used across all test modules in the PlanCritic test suite.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Tuple, Any

# Set random seeds for reproducible tests
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


@pytest.fixture(scope="session")
def device():
    """Determine the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="plancritic_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_trajectory():
    """Generate a sample trajectory for testing."""
    seq_len = 50
    dt = 0.1
    
    # Create smooth trajectory with realistic motion
    t = np.arange(seq_len) * dt
    
    # Sinusoidal path with varying speed
    x = t * 5 + np.sin(t * 0.5) * 2
    y = np.cos(t * 0.3) * 3
    
    # Compute velocities
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    
    trajectory = np.column_stack([x, y, vx, vy])
    return trajectory


@pytest.fixture
def sample_multi_agent_scenario():
    """Generate a multi-agent scenario for testing."""
    seq_len = 80
    num_agents = 5
    dt = 0.1
    
    trajectories = []
    for i in range(num_agents):
        # Each agent follows a different path
        t = np.arange(seq_len) * dt
        
        # Vary starting positions and paths
        start_x = i * 10
        start_y = (i % 2) * 6 - 3
        
        x = start_x + t * (3 + i * 0.5) + np.sin(t * (0.2 + i * 0.1)) * 1
        y = start_y + np.cos(t * (0.15 + i * 0.05)) * 2
        
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        
        trajectory = np.column_stack([x, y, vx, vy])
        trajectories.append(trajectory)
    
    return {
        'ego_trajectory': trajectories[0],
        'other_trajectories': trajectories[1:],
        'num_agents': num_agents,
        'sequence_length': seq_len,
        'dt': dt
    }


@pytest.fixture
def sample_lane_graph():
    """Generate a sample lane graph for testing."""
    num_lanes = 15
    
    # Create lane nodes with realistic features
    lane_nodes = []
    for i in range(num_lanes):
        # Lane center points along a road network
        x = (i % 5) * 20  # 5 lanes per row
        y = (i // 5) * 50  # Multiple rows
        
        node_features = [
            x, y,  # position
            np.random.uniform(-np.pi, np.pi),  # heading
            np.random.uniform(-0.1, 0.1),  # curvature
            np.random.uniform(10, 15),  # speed limit
            np.random.choice([0, 1]),  # is_intersection
            np.random.uniform(3, 4),  # lane width
            np.random.choice([0, 1, 2])  # lane type (normal, merge, exit)
        ]
        lane_nodes.append(node_features)
    
    lane_nodes = np.array(lane_nodes)
    
    # Create lane connections (edges)
    edges = []
    edge_features = []
    
    for i in range(num_lanes - 1):
        # Connect adjacent lanes
        if (i + 1) % 5 != 0:  # Not at row end
            edges.append([i, i + 1])
            edge_features.append([1.0, 0.0])  # [connection_strength, turn_angle]
        
        # Connect to next row
        if i + 5 < num_lanes:
            edges.append([i, i + 5])
            edge_features.append([0.8, 0.0])
    
    edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)
    edge_features = np.array(edge_features) if edge_features else np.empty((0, 2))
    
    return {
        'nodes': lane_nodes,
        'edges': edges,
        'edge_features': edge_features,
        'num_nodes': num_lanes,
        'num_edges': len(edges)
    }


@pytest.fixture
def sample_physics_scenario():
    """Generate a scenario for physics testing."""
    # Create collision scenario
    seq_len = 60
    dt = 0.1
    t = np.arange(seq_len) * dt
    
    # Ego vehicle moving straight
    ego_traj = np.column_stack([
        t * 10,  # x: 10 m/s
        np.zeros_like(t),  # y: straight line
        np.ones_like(t) * 10,  # vx: constant speed
        np.zeros_like(t)  # vy: no lateral motion
    ])
    
    # Other vehicle crossing path
    other_traj = np.column_stack([
        np.ones_like(t) * 30,  # x: stationary in x
        (t - 3) * 8,  # y: crossing after 3 seconds
        np.zeros_like(t),  # vx: no longitudinal motion
        np.where(t >= 3, 8, 0)  # vy: starts moving at t=3
    ])
    
    return {
        'ego_trajectory': ego_traj,
        'other_trajectory': other_traj,
        'collision_expected': True,
        'collision_time_approx': 3.75,  # seconds
        'dt': dt
    }


@pytest.fixture
def mock_model_config():
    """Standard model configuration for testing."""
    return {
        'state_dim': 32,
        'lane_dim': 64,
        'cand_dim': 64,
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'activation': 'relu',
        'use_attention': True,
        'use_layer_norm': True
    }


@pytest.fixture
def mock_training_config():
    """Standard training configuration for testing."""
    return {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 2,  # Small for testing
        'weight_decay': 1e-5,
        'gradient_clip_norm': 1.0,
        'scheduler_type': 'cosine',
        'warmup_steps': 10,
        'eval_frequency': 5,
        'save_frequency': 10,
        'early_stopping_patience': 5
    }


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    class MockDataset:
        def __init__(self, size=20):
            self.size = size
            self.data = []
            
            for i in range(size):
                scenario = {
                    'ego_trajectory': np.random.randn(50, 4),
                    'other_trajectories': [
                        np.random.randn(50, 4) for _ in range(np.random.randint(2, 5))
                    ],
                    'lane_graph': {
                        'nodes': np.random.randn(20, 8),
                        'edges': np.random.randint(0, 20, (30, 2)),
                        'edge_features': np.random.randn(30, 4)
                    },
                    'metadata': {
                        'scenario_id': f'test_scenario_{i}',
                        'timestamp': i * 1000
                    }
                }
                self.data.append(scenario)
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return MockDataset()


@pytest.fixture
def mock_physics_analyzer():
    """Create a mock physics analyzer for testing."""
    analyzer = Mock()
    
    # Mock methods with realistic return values
    analyzer.analyze_trajectory.return_value = {
        'collision_risk': 0.2,
        'comfort_score': 0.8,
        'kinematic_feasibility': True,
        'traffic_compliance': 0.9,
        'overall_score': 0.75
    }
    
    analyzer.detect_collisions.return_value = []
    analyzer.compute_comfort_metrics.return_value = {
        'max_acceleration': 2.5,
        'max_jerk': 1.8,
        'comfort_score': 0.8
    }
    
    return analyzer


@pytest.fixture
def sample_batch_data():
    """Generate a batch of data for testing."""
    batch_size = 4
    seq_len = 60
    num_agents = 6
    num_lanes = 25
    num_candidates = 3
    
    batch = {
        'agent_states': torch.randn(batch_size, num_agents, seq_len, 6),
        'lane_graph': torch.randn(batch_size, num_lanes, 64),
        'trajectory_candidates': torch.randn(batch_size, num_candidates, seq_len, 4),
        'physics_scores': torch.randn(batch_size, num_candidates),
        'rankings': torch.randint(0, num_candidates, (batch_size, num_candidates)),
        'metadata': [
            {'scenario_id': f'batch_scenario_{i}', 'timestamp': i * 1000}
            for i in range(batch_size)
        ]
    }
    
    return batch


@pytest.fixture
def evaluation_metrics():
    """Standard evaluation metrics for testing."""
    return {
        'collision_rate': 0.05,
        'comfort_score': 0.82,
        'progress_score': 0.91,
        'rule_compliance': 0.96,
        'overall_score': 0.84,
        'inference_time_ms': 15.2,
        'memory_usage_mb': 128.5
    }


# Test utilities
def assert_tensor_properties(tensor: torch.Tensor, 
                           expected_shape: Tuple[int, ...] = None,
                           expected_dtype: torch.dtype = None,
                           expected_device: torch.device = None,
                           finite_values: bool = True):
    """Assert tensor has expected properties."""
    assert isinstance(tensor, torch.Tensor)
    
    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if expected_device is not None:
        assert tensor.device == expected_device, f"Expected device {expected_device}, got {tensor.device}"
    
    if finite_values:
        assert torch.all(torch.isfinite(tensor)), "Tensor contains non-finite values"


def assert_trajectory_properties(trajectory: np.ndarray,
                               expected_length: int = None,
                               max_speed: float = 50.0,
                               max_acceleration: float = 10.0):
    """Assert trajectory has realistic physical properties."""
    assert trajectory.ndim == 2, f"Expected 2D trajectory, got {trajectory.ndim}D"
    assert trajectory.shape[1] == 4, f"Expected 4 features [x, y, vx, vy], got {trajectory.shape[1]}"
    
    if expected_length is not None:
        assert len(trajectory) == expected_length
    
    # Check speeds are reasonable
    speeds = np.sqrt(trajectory[:, 2]**2 + trajectory[:, 3]**2)
    assert np.all(speeds >= 0), "Negative speeds detected"
    assert np.all(speeds <= max_speed), f"Speeds exceed {max_speed} m/s"
    
    # Check accelerations are reasonable (if trajectory is long enough)
    if len(trajectory) > 1:
        dt = 0.1  # Assume 10 Hz
        accelerations = np.diff(trajectory[:, 2:], axis=0) / dt
        accel_magnitudes = np.sqrt(accelerations[:, 0]**2 + accelerations[:, 1]**2)
        assert np.all(accel_magnitudes <= max_acceleration), f"Accelerations exceed {max_acceleration} m/s²"


def create_collision_scenario(collision_time: float = 2.0, 
                            ego_speed: float = 10.0,
                            other_speed: float = 8.0) -> Dict[str, np.ndarray]:
    """Create a controlled collision scenario for testing."""
    dt = 0.1
    total_time = collision_time + 2.0
    t = np.arange(0, total_time, dt)
    
    # Ego vehicle moving in +x direction
    ego_x = t * ego_speed
    ego_y = np.zeros_like(t)
    ego_vx = np.ones_like(t) * ego_speed
    ego_vy = np.zeros_like(t)
    
    ego_trajectory = np.column_stack([ego_x, ego_y, ego_vx, ego_vy])
    
    # Other vehicle moving in +y direction, timed to collide
    collision_x = collision_time * ego_speed
    other_x = np.ones_like(t) * collision_x
    other_y = (t - collision_time) * other_speed
    other_vx = np.zeros_like(t)
    other_vy = np.ones_like(t) * other_speed
    
    other_trajectory = np.column_stack([other_x, other_y, other_vx, other_vy])
    
    return {
        'ego_trajectory': ego_trajectory,
        'other_trajectory': other_trajectory,
        'collision_time': collision_time,
        'collision_point': [collision_x, 0]
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


# Skip GPU tests if CUDA is not available
def pytest_runtest_setup(item):
    """Setup function to skip tests based on availability."""
    if "gpu" in [mark.name for mark in item.iter_markers()]:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")


# Custom assertions for common test patterns
class CustomAssertions:
    """Custom assertion methods for PlanCritic tests."""
    
    @staticmethod
    def assert_model_output_valid(output: torch.Tensor, 
                                batch_size: int, 
                                num_candidates: int = None):
        """Assert model output has valid shape and values."""
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size
        
        if num_candidates is not None:
            assert output.shape[1] == num_candidates
        
        # Check values are in valid range [0, 1] for scores
        assert torch.all(output >= 0), "Negative scores detected"
        assert torch.all(output <= 1), "Scores exceed 1.0"
        assert torch.all(torch.isfinite(output)), "Non-finite values detected"
    
    @staticmethod
    def assert_loss_valid(loss: torch.Tensor):
        """Assert loss value is valid."""
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.requires_grad, "Loss should require gradients"


# Make custom assertions available as pytest fixture
@pytest.fixture
def custom_assert():
    """Provide custom assertion methods."""
    return CustomAssertions()