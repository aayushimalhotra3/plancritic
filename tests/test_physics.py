"""
Unit tests for PlanCritic physics analysis module.

This module tests the physics-based trajectory analysis including:
- Collision detection algorithms
- Comfort and safety metrics
- Kinematic constraint validation
- Traffic rule compliance
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from plancritic.physics.collision import (
    CollisionDetector, check_trajectory_collisions,
    compute_safety_distances, get_collision_points
)
from plancritic.physics.comfort import (
    ComfortAnalyzer, compute_acceleration_profile,
    compute_jerk_profile, evaluate_comfort_metrics
)
from plancritic.physics.kinematics import (
    KinematicValidator, validate_trajectory_continuity,
    compute_velocity_profile, compute_curvature_profile
)
from plancritic.physics.traffic_rules import (
    TrafficRuleChecker, check_speed_limits,
    check_lane_boundaries, check_traffic_signals
)


class TestCollisionDetector:
    """Test cases for collision detection functionality."""
    
    @pytest.fixture
    def collision_detector(self):
        """Initialize collision detector with standard parameters."""
        return CollisionDetector(
            safety_margin=0.5,
            vehicle_length=4.5,
            vehicle_width=2.0,
            prediction_horizon=5.0,
            time_step=0.1
        )
    
    def test_simple_collision_detection(self, collision_detector):
        """Test basic collision detection between two trajectories."""
        # Create two intersecting trajectories
        time_steps = np.arange(0, 5.0, 0.1)
        
        # Trajectory 1: moving right
        traj1 = np.column_stack([
            time_steps * 2,  # x
            np.zeros_like(time_steps),  # y
            np.ones_like(time_steps) * 2,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        # Trajectory 2: moving up, intersecting at (4, 2) at t=2.0
        traj2 = np.column_stack([
            np.ones_like(time_steps) * 4,  # x
            time_steps * 1,  # y
            np.zeros_like(time_steps),  # vx
            np.ones_like(time_steps) * 1  # vy
        ])
        
        collision_time = collision_detector.detect_collision(traj1, traj2)
        assert collision_time is not None
        assert abs(collision_time - 2.0) < 0.2  # Should detect collision around t=2.0
    
    def test_no_collision_parallel_trajectories(self, collision_detector):
        """Test that parallel trajectories don't register collisions."""
        time_steps = np.arange(0, 3.0, 0.1)
        
        # Two parallel trajectories
        traj1 = np.column_stack([
            time_steps * 2,  # x
            np.zeros_like(time_steps),  # y = 0
            np.ones_like(time_steps) * 2,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        traj2 = np.column_stack([
            time_steps * 2,  # x
            np.ones_like(time_steps) * 5,  # y = 5 (safe distance)
            np.ones_like(time_steps) * 2,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        collision_time = collision_detector.detect_collision(traj1, traj2)
        assert collision_time is None
    
    def test_multi_agent_collision_detection(self, collision_detector):
        """Test collision detection with multiple agents."""
        # Create ego trajectory
        time_steps = np.arange(0, 4.0, 0.1)
        ego_traj = np.column_stack([
            time_steps * 3,  # x
            np.zeros_like(time_steps),  # y
            np.ones_like(time_steps) * 3,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        # Create multiple other agent trajectories
        other_trajs = []
        for i in range(3):
            other_traj = np.column_stack([
                np.ones_like(time_steps) * (6 + i * 2),  # x
                time_steps * 2,  # y
                np.zeros_like(time_steps),  # vx
                np.ones_like(time_steps) * 2  # vy
            ])
            other_trajs.append(other_traj)
        
        collisions = collision_detector.detect_multi_agent_collisions(ego_traj, other_trajs)
        assert isinstance(collisions, list)
        assert len(collisions) == len(other_trajs)
    
    def test_safety_distance_computation(self, collision_detector):
        """Test computation of safety distances based on velocities."""
        velocities = np.array([0, 5, 10, 15, 20])  # m/s
        
        safety_distances = collision_detector.compute_safety_distances(velocities)
        
        # Safety distance should increase with velocity
        assert np.all(np.diff(safety_distances) >= 0)
        assert safety_distances[0] >= collision_detector.safety_margin
    
    def test_collision_point_computation(self, collision_detector):
        """Test computation of exact collision points."""
        # Create simple head-on collision scenario
        time_steps = np.arange(0, 2.0, 0.1)
        
        traj1 = np.column_stack([
            time_steps * 5,  # x: 0 to 10
            np.zeros_like(time_steps),  # y
            np.ones_like(time_steps) * 5,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        traj2 = np.column_stack([
            10 - time_steps * 5,  # x: 10 to 0
            np.zeros_like(time_steps),  # y
            -np.ones_like(time_steps) * 5,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        collision_point = collision_detector.get_collision_point(traj1, traj2)
        assert collision_point is not None
        assert abs(collision_point[0] - 5.0) < 0.5  # Should collide around x=5
        assert abs(collision_point[1] - 0.0) < 0.1  # y should be ~0


class TestComfortAnalyzer:
    """Test cases for comfort and smoothness analysis."""
    
    @pytest.fixture
    def comfort_analyzer(self):
        """Initialize comfort analyzer with standard parameters."""
        return ComfortAnalyzer(
            max_acceleration=4.0,  # m/s²
            max_deceleration=6.0,  # m/s²
            max_jerk=3.0,  # m/s³
            max_lateral_acceleration=2.5,  # m/s²
            comfort_weights={'accel': 0.4, 'jerk': 0.3, 'lateral': 0.3}
        )
    
    def test_smooth_trajectory_comfort(self, comfort_analyzer):
        """Test comfort analysis for a smooth trajectory."""
        # Create smooth sinusoidal trajectory
        time_steps = np.arange(0, 10.0, 0.1)
        trajectory = np.column_stack([
            time_steps * 2,  # x: constant velocity
            np.sin(time_steps * 0.5) * 2,  # y: gentle sinusoid
            np.ones_like(time_steps) * 2,  # vx: constant
            np.cos(time_steps * 0.5) * 1  # vy: derivative of y
        ])
        
        comfort_score = comfort_analyzer.analyze_trajectory(trajectory)
        assert 0.0 <= comfort_score <= 1.0
        assert comfort_score > 0.7  # Should be comfortable
    
    def test_harsh_trajectory_comfort(self, comfort_analyzer):
        """Test comfort analysis for a harsh trajectory with sudden changes."""
        time_steps = np.arange(0, 5.0, 0.1)
        
        # Create trajectory with sudden direction change
        x = np.where(time_steps < 2.5, time_steps * 10, 25 - (time_steps - 2.5) * 10)
        y = np.zeros_like(time_steps)
        vx = np.where(time_steps < 2.5, 10, -10)
        vy = np.zeros_like(time_steps)
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        comfort_score = comfort_analyzer.analyze_trajectory(trajectory)
        assert 0.0 <= comfort_score <= 1.0
        assert comfort_score < 0.5  # Should be uncomfortable due to harsh change
    
    def test_acceleration_profile_computation(self, comfort_analyzer):
        """Test computation of acceleration profiles."""
        # Create trajectory with known acceleration pattern
        time_steps = np.arange(0, 4.0, 0.1)
        
        # Quadratic position -> linear velocity -> constant acceleration
        x = 0.5 * time_steps**2  # x = 0.5*a*t²
        y = np.zeros_like(time_steps)
        vx = time_steps  # vx = a*t
        vy = np.zeros_like(time_steps)
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        accel_profile = comfort_analyzer.compute_acceleration_profile(trajectory)
        
        # Should have constant longitudinal acceleration ≈ 1.0 m/s²
        assert accel_profile.shape[0] == len(time_steps) - 1  # One less due to differentiation
        assert np.allclose(accel_profile[:, 0], 1.0, atol=0.1)  # ax ≈ 1.0
        assert np.allclose(accel_profile[:, 1], 0.0, atol=0.1)  # ay ≈ 0.0
    
    def test_jerk_profile_computation(self, comfort_analyzer):
        """Test computation of jerk (acceleration derivative) profiles."""
        time_steps = np.arange(0, 3.0, 0.1)
        
        # Create trajectory with changing acceleration
        x = (1/6) * time_steps**3  # Cubic position
        y = np.zeros_like(time_steps)
        vx = 0.5 * time_steps**2  # Quadratic velocity
        vy = np.zeros_like(time_steps)
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        jerk_profile = comfort_analyzer.compute_jerk_profile(trajectory)
        
        # Should have constant jerk ≈ 1.0 m/s³
        assert jerk_profile.shape[0] == len(time_steps) - 2  # Two less due to double differentiation
        assert np.allclose(jerk_profile[:, 0], 1.0, atol=0.2)  # jx ≈ 1.0
    
    def test_lateral_acceleration_computation(self, comfort_analyzer):
        """Test computation of lateral acceleration for curved paths."""
        # Create circular trajectory
        time_steps = np.arange(0, 2*np.pi, 0.1)
        radius = 10.0
        angular_velocity = 1.0
        
        x = radius * np.cos(angular_velocity * time_steps)
        y = radius * np.sin(angular_velocity * time_steps)
        vx = -radius * angular_velocity * np.sin(angular_velocity * time_steps)
        vy = radius * angular_velocity * np.cos(angular_velocity * time_steps)
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        lateral_accel = comfort_analyzer.compute_lateral_acceleration(trajectory)
        
        # For circular motion: a_lateral = v²/r = (r*ω)²/r = r*ω²
        expected_lateral_accel = radius * angular_velocity**2
        assert np.allclose(np.abs(lateral_accel), expected_lateral_accel, atol=0.5)


class TestKinematicValidator:
    """Test cases for kinematic constraint validation."""
    
    @pytest.fixture
    def kinematic_validator(self):
        """Initialize kinematic validator with vehicle parameters."""
        return KinematicValidator(
            max_speed=30.0,  # m/s
            max_acceleration=5.0,  # m/s²
            max_deceleration=8.0,  # m/s²
            max_steering_angle=np.pi/4,  # 45 degrees
            wheelbase=2.8,  # m
            dt=0.1  # time step
        )
    
    def test_valid_trajectory_validation(self, kinematic_validator):
        """Test validation of a kinematically feasible trajectory."""
        # Create smooth, feasible trajectory
        time_steps = np.arange(0, 5.0, 0.1)
        
        x = time_steps * 5  # Constant 5 m/s
        y = np.zeros_like(time_steps)
        vx = np.ones_like(time_steps) * 5
        vy = np.zeros_like(time_steps)
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        is_valid, violations = kinematic_validator.validate_trajectory(trajectory)
        assert is_valid
        assert len(violations) == 0
    
    def test_speed_limit_violation(self, kinematic_validator):
        """Test detection of speed limit violations."""
        time_steps = np.arange(0, 3.0, 0.1)
        
        # Trajectory exceeding speed limit
        x = time_steps * 35  # 35 m/s > 30 m/s limit
        y = np.zeros_like(time_steps)
        vx = np.ones_like(time_steps) * 35
        vy = np.zeros_like(time_steps)
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        is_valid, violations = kinematic_validator.validate_trajectory(trajectory)
        assert not is_valid
        assert any('speed' in v.lower() for v in violations)
    
    def test_acceleration_limit_violation(self, kinematic_validator):
        """Test detection of acceleration limit violations."""
        time_steps = np.arange(0, 2.0, 0.1)
        
        # Trajectory with excessive acceleration
        vx = time_steps * 10  # 10 m/s² acceleration > 5 m/s² limit
        x = 0.5 * 10 * time_steps**2  # Integrate velocity
        y = np.zeros_like(time_steps)
        vy = np.zeros_like(time_steps)
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        is_valid, violations = kinematic_validator.validate_trajectory(trajectory)
        assert not is_valid
        assert any('acceleration' in v.lower() for v in violations)
    
    def test_trajectory_continuity_validation(self, kinematic_validator):
        """Test validation of trajectory continuity."""
        # Create trajectory with discontinuity
        time_steps = np.arange(0, 4.0, 0.1)
        
        x = time_steps * 2
        y = np.zeros_like(time_steps)
        vx = np.ones_like(time_steps) * 2
        vy = np.zeros_like(time_steps)
        
        # Introduce discontinuity at midpoint
        mid_idx = len(time_steps) // 2
        x[mid_idx:] += 10  # Jump in position
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        is_continuous = kinematic_validator.check_continuity(trajectory)
        assert not is_continuous
    
    def test_curvature_computation(self, kinematic_validator):
        """Test computation of trajectory curvature."""
        # Create circular arc
        time_steps = np.arange(0, np.pi, 0.1)
        radius = 5.0
        
        x = radius * np.cos(time_steps)
        y = radius * np.sin(time_steps)
        vx = -radius * np.sin(time_steps) * 1.0  # Angular velocity = 1
        vy = radius * np.cos(time_steps) * 1.0
        
        trajectory = np.column_stack([x, y, vx, vy])
        
        curvature = kinematic_validator.compute_curvature(trajectory)
        
        # For a circle, curvature = 1/radius
        expected_curvature = 1.0 / radius
        assert np.allclose(np.abs(curvature), expected_curvature, atol=0.1)


class TestTrafficRuleChecker:
    """Test cases for traffic rule compliance checking."""
    
    @pytest.fixture
    def traffic_checker(self):
        """Initialize traffic rule checker."""
        return TrafficRuleChecker(
            speed_limit_tolerance=2.0,  # m/s
            lane_boundary_tolerance=0.3,  # m
            signal_compliance_distance=50.0  # m
        )
    
    @pytest.fixture
    def sample_lane_boundaries(self):
        """Create sample lane boundary data."""
        return {
            'left_boundary': np.array([[0, 2], [100, 2]]),  # Left lane boundary
            'right_boundary': np.array([[0, -2], [100, -2]]),  # Right lane boundary
            'center_line': np.array([[0, 0], [100, 0]])  # Lane center
        }
    
    def test_speed_limit_compliance(self, traffic_checker):
        """Test speed limit compliance checking."""
        # Create trajectory within speed limits
        time_steps = np.arange(0, 5.0, 0.1)
        speed_limit = 15.0  # m/s
        
        trajectory = np.column_stack([
            time_steps * 12,  # x: 12 m/s < 15 m/s limit
            np.zeros_like(time_steps),  # y
            np.ones_like(time_steps) * 12,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        violations = traffic_checker.check_speed_limits(trajectory, speed_limit)
        assert len(violations) == 0
        
        # Test with speed limit violation
        trajectory_fast = np.column_stack([
            time_steps * 20,  # x: 20 m/s > 15 m/s limit
            np.zeros_like(time_steps),  # y
            np.ones_like(time_steps) * 20,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        violations_fast = traffic_checker.check_speed_limits(trajectory_fast, speed_limit)
        assert len(violations_fast) > 0
    
    def test_lane_boundary_compliance(self, traffic_checker, sample_lane_boundaries):
        """Test lane boundary compliance checking."""
        time_steps = np.arange(0, 10.0, 0.1)
        
        # Trajectory staying within lane boundaries
        trajectory_valid = np.column_stack([
            time_steps * 5,  # x
            np.zeros_like(time_steps),  # y: staying at center
            np.ones_like(time_steps) * 5,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        violations = traffic_checker.check_lane_boundaries(
            trajectory_valid, sample_lane_boundaries
        )
        assert len(violations) == 0
        
        # Trajectory crossing lane boundary
        trajectory_crossing = np.column_stack([
            time_steps * 5,  # x
            np.ones_like(time_steps) * 3,  # y: outside lane (>2m)
            np.ones_like(time_steps) * 5,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        violations_crossing = traffic_checker.check_lane_boundaries(
            trajectory_crossing, sample_lane_boundaries
        )
        assert len(violations_crossing) > 0
    
    def test_traffic_signal_compliance(self, traffic_checker):
        """Test traffic signal compliance checking."""
        # Create trajectory approaching red light
        time_steps = np.arange(0, 8.0, 0.1)
        
        trajectory = np.column_stack([
            time_steps * 10,  # x: approaching at 10 m/s
            np.zeros_like(time_steps),  # y
            np.ones_like(time_steps) * 10,  # vx
            np.zeros_like(time_steps)  # vy
        ])
        
        # Red light at x=60m
        traffic_signals = [
            {'position': [60, 0], 'state': 'red', 'stop_line': 58}
        ]
        
        violations = traffic_checker.check_traffic_signals(trajectory, traffic_signals)
        
        # Should detect red light violation if trajectory doesn't stop
        assert len(violations) > 0
    
    def test_right_of_way_compliance(self, traffic_checker):
        """Test right-of-way rule compliance."""
        # Create intersection scenario
        ego_trajectory = np.column_stack([
            np.arange(0, 50, 1),  # x: straight through
            np.zeros(50),  # y
            np.ones(50),  # vx
            np.zeros(50)  # vy
        ])
        
        # Other vehicle with right of way
        other_trajectory = np.column_stack([
            np.ones(50) * 25,  # x: crossing point
            np.arange(-25, 25, 1),  # y: crossing path
            np.zeros(50),  # vx
            np.ones(50)  # vy
        ])
        
        violations = traffic_checker.check_right_of_way(
            ego_trajectory, [other_trajectory], intersection_point=[25, 0]
        )
        
        # Should detect potential right-of-way violation
        assert isinstance(violations, list)


class TestPhysicsIntegration:
    """Integration tests for complete physics analysis pipeline."""
    
    def test_complete_physics_analysis(self):
        """Test complete physics analysis of a trajectory."""
        from plancritic.physics.analyzer import PhysicsAnalyzer
        
        # Initialize complete physics analyzer
        analyzer = PhysicsAnalyzer(
            collision_detector=CollisionDetector(),
            comfort_analyzer=ComfortAnalyzer(),
            kinematic_validator=KinematicValidator(),
            traffic_checker=TrafficRuleChecker()
        )
        
        # Create sample scenario
        time_steps = np.arange(0, 8.0, 0.1)
        ego_trajectory = np.column_stack([
            time_steps * 8,  # x
            np.sin(time_steps * 0.5) * 1,  # y: gentle curve
            np.ones_like(time_steps) * 8,  # vx
            np.cos(time_steps * 0.5) * 0.5  # vy
        ])
        
        other_trajectories = [
            np.column_stack([
                np.ones_like(time_steps) * 30,  # x: stationary
                np.ones_like(time_steps) * 5,  # y: parallel lane
                np.zeros_like(time_steps),  # vx
                np.zeros_like(time_steps)  # vy
            ])
        ]
        
        # Run complete analysis
        results = analyzer.analyze_trajectory(
            ego_trajectory, 
            other_trajectories,
            speed_limit=15.0,
            lane_boundaries={'left': [[0, 3], [100, 3]], 'right': [[0, -3], [100, -3]]}
        )
        
        # Verify analysis results structure
        assert 'collision_risk' in results
        assert 'comfort_score' in results
        assert 'kinematic_feasibility' in results
        assert 'traffic_compliance' in results
        assert 'overall_score' in results
        
        # Verify score ranges
        assert 0.0 <= results['collision_risk'] <= 1.0
        assert 0.0 <= results['comfort_score'] <= 1.0
        assert 0.0 <= results['overall_score'] <= 1.0
    
    def test_batch_trajectory_analysis(self):
        """Test physics analysis for multiple trajectory candidates."""
        from plancritic.physics.analyzer import PhysicsAnalyzer
        
        analyzer = PhysicsAnalyzer()
        
        # Create multiple trajectory candidates
        time_steps = np.arange(0, 5.0, 0.1)
        candidates = []
        
        for i in range(3):
            trajectory = np.column_stack([
                time_steps * (5 + i),  # Different speeds
                np.sin(time_steps * (0.3 + i * 0.2)) * (1 + i * 0.5),  # Different curves
                np.ones_like(time_steps) * (5 + i),  # vx
                np.cos(time_steps * (0.3 + i * 0.2)) * (0.3 + i * 0.2)  # vy
            ])
            candidates.append(trajectory)
        
        # Analyze all candidates
        results = analyzer.analyze_batch(candidates, other_trajectories=[])
        
        assert len(results) == len(candidates)
        for result in results:
            assert 'overall_score' in result
            assert 0.0 <= result['overall_score'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])