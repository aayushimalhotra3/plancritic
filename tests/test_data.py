"""
Unit tests for PlanCritic data processing module.

This module tests the data handling functionality including:
- Dataset loading and preprocessing
- Trajectory data generation and augmentation
- Feature extraction and normalization
- Data pipeline integration
"""

import pytest
import numpy as np
import torch
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from plancritic.data.datasets import (
    TrajectoryDataset, NuPlanDataset, WaymoDataset,
    SyntheticDataset, load_dataset
)
from plancritic.data.preprocessing import (
    TrajectoryPreprocessor, normalize_trajectories,
    augment_trajectories, extract_features
)
from plancritic.data.generators import (
    SyntheticTrajectoryGenerator, ScenarioGenerator,
    generate_lane_graph, generate_traffic_scenario
)
from plancritic.data.utils import (
    trajectory_to_tensor, batch_trajectories,
    compute_trajectory_statistics, validate_trajectory_format
)


class TestTrajectoryDataset:
    """Test cases for trajectory dataset handling."""
    
    @pytest.fixture
    def sample_trajectory_data(self):
        """Create sample trajectory data for testing."""
        num_scenarios = 10
        seq_len = 50
        
        data = []
        for i in range(num_scenarios):
            scenario = {
                'ego_trajectory': np.random.randn(seq_len, 4),  # [x, y, vx, vy]
                'other_trajectories': [
                    np.random.randn(seq_len, 4) for _ in range(np.random.randint(2, 6))
                ],
                'lane_graph': {
                    'nodes': np.random.randn(20, 8),  # lane node features
                    'edges': np.random.randint(0, 20, (40, 2)),  # edge connections
                    'edge_features': np.random.randn(40, 4)
                },
                'metadata': {
                    'scenario_id': f'scenario_{i}',
                    'timestamp': i * 1000,
                    'weather': 'clear',
                    'time_of_day': 'day'
                }
            }
            data.append(scenario)
        
        return data
    
    @pytest.fixture
    def trajectory_dataset(self, sample_trajectory_data):
        """Initialize trajectory dataset with sample data."""
        return TrajectoryDataset(sample_trajectory_data)
    
    def test_dataset_initialization(self, trajectory_dataset, sample_trajectory_data):
        """Test dataset initialization and basic properties."""
        assert len(trajectory_dataset) == len(sample_trajectory_data)
        assert trajectory_dataset.num_scenarios == len(sample_trajectory_data)
        
        # Test dataset indexing
        sample = trajectory_dataset[0]
        assert 'ego_trajectory' in sample
        assert 'other_trajectories' in sample
        assert 'lane_graph' in sample
        assert 'metadata' in sample
    
    def test_dataset_iteration(self, trajectory_dataset):
        """Test dataset iteration and batching."""
        # Test iteration
        for i, sample in enumerate(trajectory_dataset):
            assert isinstance(sample, dict)
            assert 'ego_trajectory' in sample
            if i >= 2:  # Test first few samples
                break
        
        # Test with DataLoader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(trajectory_dataset, batch_size=3, shuffle=False)
        batch = next(iter(dataloader))
        
        assert 'ego_trajectory' in batch
        assert batch['ego_trajectory'].shape[0] == 3  # batch size
    
    def test_dataset_filtering(self, trajectory_dataset):
        """Test dataset filtering by criteria."""
        # Filter by scenario length
        filtered = trajectory_dataset.filter_by_length(min_length=40)
        assert len(filtered) <= len(trajectory_dataset)
        
        # Filter by metadata
        filtered_weather = trajectory_dataset.filter_by_metadata('weather', 'clear')
        assert len(filtered_weather) <= len(trajectory_dataset)
    
    def test_dataset_statistics(self, trajectory_dataset):
        """Test computation of dataset statistics."""
        stats = trajectory_dataset.compute_statistics()
        
        assert 'num_scenarios' in stats
        assert 'avg_trajectory_length' in stats
        assert 'avg_num_agents' in stats
        assert 'velocity_stats' in stats
        
        # Verify statistics are reasonable
        assert stats['num_scenarios'] == len(trajectory_dataset)
        assert stats['avg_trajectory_length'] > 0
        assert stats['avg_num_agents'] > 0


class TestNuPlanDataset:
    """Test cases for NuPlan dataset integration."""
    
    @pytest.fixture
    def mock_nuplan_data(self):
        """Mock NuPlan data structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data files
            scenarios_file = os.path.join(temp_dir, 'scenarios.pkl')
            maps_file = os.path.join(temp_dir, 'maps.pkl')
            
            # Mock scenario data
            scenarios = {
                'scenario_1': {
                    'ego_states': np.random.randn(100, 7),  # [x, y, heading, vx, vy, ax, ay]
                    'agent_states': {
                        'agent_1': np.random.randn(100, 7),
                        'agent_2': np.random.randn(100, 7)
                    },
                    'map_data': 'map_1'
                }
            }
            
            maps = {
                'map_1': {
                    'lane_segments': np.random.randn(50, 10),
                    'traffic_lights': np.random.randn(10, 5)
                }
            }
            
            # Save mock data (in real implementation, would use pickle)
            yield temp_dir, scenarios, maps
    
    @patch('plancritic.data.datasets.load_nuplan_scenarios')
    def test_nuplan_dataset_loading(self, mock_load, mock_nuplan_data):
        """Test NuPlan dataset loading and preprocessing."""
        temp_dir, scenarios, maps = mock_nuplan_data
        mock_load.return_value = scenarios, maps
        
        dataset = NuPlanDataset(data_root=temp_dir, split='train')
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert 'ego_trajectory' in sample
        assert 'lane_graph' in sample
    
    def test_nuplan_coordinate_transformation(self):
        """Test coordinate system transformations for NuPlan data."""
        # Mock global coordinates
        global_coords = np.array([[100, 200, np.pi/4], [105, 205, np.pi/4]])
        
        # Transform to ego-centric coordinates
        ego_coords = NuPlanDataset.transform_to_ego_frame(
            global_coords, ego_pose=[100, 200, np.pi/4]
        )
        
        # First point should be at origin
        assert np.allclose(ego_coords[0, :2], [0, 0], atol=1e-6)
        assert np.allclose(ego_coords[0, 2], 0, atol=1e-6)


class TestWaymoDataset:
    """Test cases for Waymo dataset integration."""
    
    @pytest.fixture
    def mock_waymo_data(self):
        """Mock Waymo dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock TFRecord files
            tfrecord_file = os.path.join(temp_dir, 'scenario_001.tfrecord')
            
            # In real implementation, would create actual TFRecord
            # For testing, just create empty file
            with open(tfrecord_file, 'wb') as f:
                f.write(b'mock_tfrecord_data')
            
            yield temp_dir
    
    @patch('plancritic.data.datasets.parse_waymo_scenario')
    def test_waymo_dataset_loading(self, mock_parse, mock_waymo_data):
        """Test Waymo dataset loading and parsing."""
        mock_parse.return_value = {
            'ego_trajectory': np.random.randn(91, 4),  # Waymo uses 91 timesteps
            'other_trajectories': [np.random.randn(91, 4) for _ in range(5)],
            'lane_graph': {'nodes': np.random.randn(30, 6), 'edges': np.random.randint(0, 30, (60, 2))},
            'scenario_id': 'waymo_scenario_001'
        }
        
        dataset = WaymoDataset(data_root=mock_waymo_data, split='validation')
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert sample['ego_trajectory'].shape[0] == 91  # Waymo timesteps
    
    def test_waymo_feature_extraction(self):
        """Test feature extraction from Waymo data format."""
        # Mock Waymo track data
        track_data = {
            'states': np.random.randn(91, 7),  # [x, y, z, length, width, height, heading]
            'valid': np.ones(91, dtype=bool),
            'object_type': 1  # Vehicle
        }
        
        features = WaymoDataset.extract_track_features(track_data)
        
        assert 'trajectory' in features
        assert 'object_type' in features
        assert 'valid_mask' in features
        assert features['trajectory'].shape == (91, 4)  # [x, y, vx, vy]


class TestSyntheticDataset:
    """Test cases for synthetic data generation."""
    
    @pytest.fixture
    def synthetic_dataset(self):
        """Initialize synthetic dataset generator."""
        return SyntheticDataset(
            num_scenarios=100,
            scenario_length=80,
            num_agents_range=(3, 8),
            map_size=(200, 200),
            seed=42
        )
    
    def test_synthetic_data_generation(self, synthetic_dataset):
        """Test synthetic trajectory generation."""
        assert len(synthetic_dataset) == 100
        
        sample = synthetic_dataset[0]
        assert 'ego_trajectory' in sample
        assert 'other_trajectories' in sample
        assert 'lane_graph' in sample
        
        # Check trajectory properties
        ego_traj = sample['ego_trajectory']
        assert ego_traj.shape == (80, 4)  # [x, y, vx, vy]
        
        # Check that trajectories are physically reasonable
        velocities = np.sqrt(ego_traj[:, 2]**2 + ego_traj[:, 3]**2)
        assert np.all(velocities >= 0)  # Non-negative speeds
        assert np.all(velocities <= 50)  # Reasonable speed limits
    
    def test_synthetic_scenario_diversity(self, synthetic_dataset):
        """Test diversity of generated synthetic scenarios."""
        scenarios = [synthetic_dataset[i] for i in range(10)]
        
        # Check that scenarios are different
        ego_trajectories = [s['ego_trajectory'] for s in scenarios]
        
        for i in range(len(ego_trajectories)):
            for j in range(i + 1, len(ego_trajectories)):
                # Trajectories should not be identical
                assert not np.allclose(ego_trajectories[i], ego_trajectories[j])
    
    def test_synthetic_lane_graph_generation(self, synthetic_dataset):
        """Test synthetic lane graph generation."""
        sample = synthetic_dataset[0]
        lane_graph = sample['lane_graph']
        
        assert 'nodes' in lane_graph
        assert 'edges' in lane_graph
        
        nodes = lane_graph['nodes']
        edges = lane_graph['edges']
        
        assert nodes.shape[1] >= 4  # At least [x, y, heading, curvature]
        assert edges.shape[1] == 2  # [source, target]
        assert np.all(edges >= 0)  # Valid node indices
        assert np.all(edges < len(nodes))  # Within bounds


class TestTrajectoryPreprocessor:
    """Test cases for trajectory preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        """Initialize trajectory preprocessor."""
        return TrajectoryPreprocessor(
            normalize_coordinates=True,
            resample_rate=10,  # Hz
            filter_stationary=True,
            min_displacement=1.0,  # meters
            coordinate_frame='ego'
        )
    
    def test_coordinate_normalization(self, preprocessor):
        """Test coordinate normalization to ego frame."""
        # Create trajectory in global coordinates
        global_traj = np.array([
            [100, 200, 5, 0],  # ego position
            [105, 200, 5, 0],
            [110, 205, 4, 2],
            [114, 209, 3, 3]
        ])
        
        ego_pose = [100, 200, 0]  # [x, y, heading]
        
        normalized = preprocessor.normalize_to_ego_frame(global_traj, ego_pose)
        
        # First point should be at origin
        assert np.allclose(normalized[0, :2], [0, 0], atol=1e-6)
    
    def test_trajectory_resampling(self, preprocessor):
        """Test trajectory resampling to uniform time intervals."""
        # Create irregular time sampling
        timestamps = np.array([0, 0.05, 0.15, 0.25, 0.4, 0.5])
        trajectory = np.random.randn(len(timestamps), 4)
        
        resampled = preprocessor.resample_trajectory(trajectory, timestamps, dt=0.1)
        
        # Should have uniform 0.1s intervals
        expected_length = int((timestamps[-1] - timestamps[0]) / 0.1) + 1
        assert resampled.shape[0] == expected_length
    
    def test_stationary_filtering(self, preprocessor):
        """Test filtering of stationary trajectories."""
        # Create mostly stationary trajectory
        stationary_traj = np.array([
            [0, 0, 0, 0],
            [0.1, 0, 0.1, 0],
            [0.1, 0.1, 0, 0.1],
            [0.2, 0.1, 0.1, 0]
        ])
        
        is_dynamic = preprocessor.filter_stationary_trajectory(stationary_traj)
        assert not is_dynamic  # Should be classified as stationary
        
        # Create dynamic trajectory
        dynamic_traj = np.array([
            [0, 0, 5, 0],
            [5, 0, 5, 0],
            [10, 0, 5, 0],
            [15, 0, 5, 0]
        ])
        
        is_dynamic = preprocessor.filter_stationary_trajectory(dynamic_traj)
        assert is_dynamic  # Should be classified as dynamic
    
    def test_trajectory_smoothing(self, preprocessor):
        """Test trajectory smoothing and noise reduction."""
        # Create noisy trajectory
        clean_traj = np.column_stack([
            np.linspace(0, 10, 50),  # x
            np.zeros(50),  # y
            np.ones(50) * 2,  # vx
            np.zeros(50)  # vy
        ])
        
        # Add noise
        noise = np.random.randn(*clean_traj.shape) * 0.1
        noisy_traj = clean_traj + noise
        
        smoothed = preprocessor.smooth_trajectory(noisy_traj, window_size=5)
        
        # Smoothed trajectory should be closer to clean trajectory
        clean_error = np.mean(np.abs(clean_traj - noisy_traj))
        smoothed_error = np.mean(np.abs(clean_traj - smoothed))
        assert smoothed_error < clean_error
    
    def test_feature_extraction(self, preprocessor):
        """Test extraction of trajectory features."""
        trajectory = np.array([
            [0, 0, 2, 0],
            [2, 0, 2, 0],
            [4, 1, 1.8, 0.5],
            [5.8, 1.5, 1.5, 0.3]
        ])
        
        features = preprocessor.extract_features(trajectory)
        
        assert 'speed' in features
        assert 'acceleration' in features
        assert 'heading' in features
        assert 'curvature' in features
        
        # Check feature shapes
        assert len(features['speed']) == len(trajectory)
        assert len(features['acceleration']) == len(trajectory) - 1


class TestDataAugmentation:
    """Test cases for data augmentation techniques."""
    
    def test_trajectory_rotation(self):
        """Test trajectory rotation augmentation."""
        trajectory = np.array([
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [2, 0, 1, 0]
        ])
        
        # Rotate by 90 degrees
        rotated = augment_trajectories(trajectory, rotation_angle=np.pi/2)
        
        # After 90° rotation, x becomes -y and y becomes x
        expected = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 2, 0, 1]
        ])
        
        assert np.allclose(rotated, expected, atol=1e-6)
    
    def test_trajectory_translation(self):
        """Test trajectory translation augmentation."""
        trajectory = np.array([
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [2, 0, 1, 0]
        ])
        
        translation = [5, 3]
        translated = augment_trajectories(trajectory, translation=translation)
        
        # Positions should be shifted, velocities unchanged
        assert np.allclose(translated[:, :2], trajectory[:, :2] + translation)
        assert np.allclose(translated[:, 2:], trajectory[:, 2:])
    
    def test_velocity_scaling(self):
        """Test velocity scaling augmentation."""
        trajectory = np.array([
            [0, 0, 2, 1],
            [2, 1, 2, 1],
            [4, 2, 2, 1]
        ])
        
        scale_factor = 1.5
        scaled = augment_trajectories(trajectory, velocity_scale=scale_factor)
        
        # Velocities should be scaled
        assert np.allclose(scaled[:, 2:], trajectory[:, 2:] * scale_factor)
    
    def test_noise_injection(self):
        """Test noise injection augmentation."""
        trajectory = np.array([
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [2, 0, 1, 0]
        ])
        
        noise_std = 0.1
        noisy = augment_trajectories(trajectory, noise_std=noise_std)
        
        # Should be different from original but similar
        assert not np.allclose(noisy, trajectory)
        assert np.mean(np.abs(noisy - trajectory)) < noise_std * 3  # Within 3 sigma


class TestDataUtils:
    """Test cases for data utility functions."""
    
    def test_trajectory_to_tensor_conversion(self):
        """Test conversion of trajectory data to PyTorch tensors."""
        trajectory = np.random.randn(50, 4)
        
        tensor = trajectory_to_tensor(trajectory)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == trajectory.shape
        assert torch.allclose(tensor, torch.from_numpy(trajectory).float())
    
    def test_batch_trajectories(self):
        """Test batching of variable-length trajectories."""
        trajectories = [
            np.random.randn(30, 4),
            np.random.randn(45, 4),
            np.random.randn(60, 4)
        ]
        
        batched, lengths = batch_trajectories(trajectories, pad_value=0.0)
        
        assert batched.shape == (3, 60, 4)  # [batch, max_len, features]
        assert lengths == [30, 45, 60]
        
        # Check padding
        assert torch.allclose(batched[0, 30:], torch.zeros(30, 4))
        assert torch.allclose(batched[1, 45:], torch.zeros(15, 4))
    
    def test_trajectory_statistics(self):
        """Test computation of trajectory statistics."""
        trajectories = [
            np.random.randn(40, 4),
            np.random.randn(50, 4),
            np.random.randn(35, 4)
        ]
        
        stats = compute_trajectory_statistics(trajectories)
        
        assert 'mean_length' in stats
        assert 'std_length' in stats
        assert 'mean_speed' in stats
        assert 'max_speed' in stats
        assert 'mean_acceleration' in stats
        
        assert stats['mean_length'] == np.mean([40, 50, 35])
    
    def test_trajectory_format_validation(self):
        """Test validation of trajectory data format."""
        # Valid trajectory
        valid_traj = np.random.randn(50, 4)
        assert validate_trajectory_format(valid_traj)
        
        # Invalid shapes
        assert not validate_trajectory_format(np.random.randn(50, 3))  # Wrong feature dim
        assert not validate_trajectory_format(np.random.randn(50))  # Wrong dimensions
        
        # Invalid values
        invalid_traj = np.random.randn(50, 4)
        invalid_traj[10, 0] = np.inf  # Infinite value
        assert not validate_trajectory_format(invalid_traj)


class TestDataPipeline:
    """Integration tests for complete data pipeline."""
    
    def test_end_to_end_data_pipeline(self):
        """Test complete data loading and preprocessing pipeline."""
        # Create synthetic dataset
        dataset = SyntheticDataset(num_scenarios=20, scenario_length=60)
        
        # Initialize preprocessor
        preprocessor = TrajectoryPreprocessor()
        
        # Process dataset
        processed_data = []
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Preprocess trajectories
            ego_traj = preprocessor.process_trajectory(sample['ego_trajectory'])
            other_trajs = [
                preprocessor.process_trajectory(traj) 
                for traj in sample['other_trajectories']
            ]
            
            processed_sample = {
                'ego_trajectory': ego_traj,
                'other_trajectories': other_trajs,
                'lane_graph': sample['lane_graph'],
                'metadata': sample['metadata']
            }
            processed_data.append(processed_sample)
        
        assert len(processed_data) == len(dataset)
        
        # Test batching
        from torch.utils.data import DataLoader
        processed_dataset = TrajectoryDataset(processed_data)
        dataloader = DataLoader(processed_dataset, batch_size=4, shuffle=True)
        
        batch = next(iter(dataloader))
        assert 'ego_trajectory' in batch
        assert batch['ego_trajectory'].shape[0] == 4  # batch size
    
    def test_data_pipeline_performance(self):
        """Test data pipeline performance and memory usage."""
        import time
        
        # Create larger dataset for performance testing
        dataset = SyntheticDataset(num_scenarios=100, scenario_length=80)
        
        start_time = time.time()
        
        # Process first 50 samples
        for i in range(50):
            sample = dataset[i]
            # Simulate processing
            _ = sample['ego_trajectory'].copy()
        
        processing_time = time.time() - start_time
        
        # Should process reasonably quickly (< 1 second for 50 samples)
        assert processing_time < 1.0
        
        # Test memory efficiency
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load and process data
        large_dataset = SyntheticDataset(num_scenarios=200, scenario_length=100)
        samples = [large_dataset[i] for i in range(100)]
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 500 MB for test data)
        assert memory_increase < 500


if __name__ == "__main__":
    pytest.main([__file__])