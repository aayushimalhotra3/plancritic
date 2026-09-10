"""Tests for the synthetic scenario generator."""

import numpy as np
import pytest

from plancritic.data.synthetic import SyntheticScenarioGenerator
from plancritic.data.samplers import SceneData, TrajectoryCandidate


class TestDeterminism:
    """Same seed + split must produce identical scenarios and labels."""

    def test_same_seed_same_output(self):
        gen1 = SyntheticScenarioGenerator(seed=7, split="train", num_scenarios=20)
        gen2 = SyntheticScenarioGenerator(seed=7, split="train", num_scenarios=20)

        scenes1 = gen1.generate()
        scenes2 = gen2.generate()

        assert len(scenes1) == len(scenes2)
        for s1, s2 in zip(scenes1, scenes2):
            assert s1.scene_id == s2.scene_id
            np.testing.assert_array_equal(s1.ego_state, s2.ego_state)
            np.testing.assert_array_equal(s1.agent_states, s2.agent_states)
            for c1, c2 in zip(s1.candidates, s2.candidates):
                np.testing.assert_array_equal(c1.waypoints, c2.waypoints)

    def test_labels_deterministic(self):
        gen1 = SyntheticScenarioGenerator(seed=7, split="val", num_scenarios=10)
        gen2 = SyntheticScenarioGenerator(seed=7, split="val", num_scenarios=10)

        r1 = gen1.generate_with_labels()
        r2 = gen2.generate_with_labels()

        for (_, l1), (_, l2) in zip(r1, r2):
            for d1, d2 in zip(l1, l2):
                assert d1["risk"] == d2["risk"]
                assert d1["comfort"] == d2["comfort"]
                assert d1["progress"] == d2["progress"]
                assert d1["collided"] == d2["collided"]


class TestSplitDisjoint:
    """Different splits must use different seed ranges."""

    def test_splits_produce_different_scenes(self):
        kwargs = dict(seed=42, num_scenarios=10)
        train = SyntheticScenarioGenerator(split="train", **kwargs).generate()
        val = SyntheticScenarioGenerator(split="val", **kwargs).generate()
        test = SyntheticScenarioGenerator(split="test", **kwargs).generate()

        # scene ids should differ
        train_ids = {s.scene_id for s in train}
        val_ids = {s.scene_id for s in val}
        test_ids = {s.scene_id for s in test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

        # ego states should differ (different seeds)
        assert not np.array_equal(train[0].ego_state, val[0].ego_state)

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError):
            SyntheticScenarioGenerator(split="foo")


class TestSceneShapes:
    """Generated data must match SceneData / TrajectoryCandidate contracts."""

    @pytest.fixture
    def scenes(self):
        gen = SyntheticScenarioGenerator(seed=0, split="train", num_scenarios=16,
                                         num_candidates=6, horizon=50)
        return gen.generate()

    def test_ego_state_shape(self, scenes):
        for s in scenes:
            assert s.ego_state.shape == (8,)

    def test_agent_states_shape(self, scenes):
        for s in scenes:
            assert s.agent_states.ndim == 2
            assert s.agent_states.shape[1] == 8
            assert s.agent_mask.shape[0] == s.agent_states.shape[0]

    def test_candidate_count(self, scenes):
        for s in scenes:
            assert len(s.candidates) == 6

    def test_waypoint_shape(self, scenes):
        for s in scenes:
            for c in s.candidates:
                assert c.waypoints.shape == (50, 4)
                assert c.timestamps.shape == (50,)

    def test_route_waypoints(self, scenes):
        for s in scenes:
            assert s.route_waypoints.ndim == 2
            assert s.route_waypoints.shape[1] == 2
            assert len(s.route_waypoints) >= 2


class TestLabelSpread:
    """Labels should have real spread, not all zeros/ones."""

    def test_risk_has_spread(self):
        gen = SyntheticScenarioGenerator(seed=0, split="train", num_scenarios=40)
        results = gen.generate_with_labels()
        risks = [d["risk"] for _, labels in results for d in labels]
        assert min(risks) < 0.3, "Expected some low-risk candidates"
        assert max(risks) > 0.3, "Expected some higher-risk candidates"

    def test_comfort_has_spread(self):
        gen = SyntheticScenarioGenerator(seed=0, split="train", num_scenarios=40)
        results = gen.generate_with_labels()
        comforts = [d["comfort"] for _, labels in results for d in labels]
        assert min(comforts) < 0.9, "Expected some uncomfortable candidates"
        assert max(comforts) > 0.5, "Expected some comfortable candidates"

    def test_progress_has_spread(self):
        gen = SyntheticScenarioGenerator(seed=0, split="train", num_scenarios=40)
        results = gen.generate_with_labels()
        progs = [d["progress"] for _, labels in results for d in labels]
        assert min(progs) < 0.5, "Expected some low-progress candidates"
        assert max(progs) > 0.5, "Expected some good-progress candidates"

    def test_collided_flag_has_both_values(self):
        gen = SyntheticScenarioGenerator(seed=0, split="train", num_scenarios=80)
        results = gen.generate_with_labels()
        flags = [d["collided"] for _, labels in results for d in labels]
        assert any(flags), "Expected at least one collision"
        assert not all(flags), "Expected some non-collisions"


class TestCollidedFlag:
    """The binary collided flag should be consistent with scenario setup."""

    def test_no_agents_means_no_collision(self):
        gen = SyntheticScenarioGenerator(seed=0, split="train", num_scenarios=4)
        results = gen.generate_with_labels()
        # first scenario is straight_empty (idx 0 % 4 == 0)
        _, labels = results[0]
        for d in labels:
            assert d["collided"] is False

    def test_collided_is_bool(self):
        gen = SyntheticScenarioGenerator(seed=0, split="train", num_scenarios=8)
        results = gen.generate_with_labels()
        for _, labels in results:
            for d in labels:
                assert isinstance(d["collided"], bool)
