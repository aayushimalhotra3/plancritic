"""
Synthetic scenario generator for training and evaluating trajectory critics.

Generates deterministic driving scenarios with seed control, producing
SceneData objects with physics-based pseudo-labels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .samplers import TrajectoryCandidate, SceneData
from ..eval.physics_checks import PhysicsChecker, PhysicsConfig


# Disjoint seed offsets per split so train/val/test never overlap.
_SPLIT_OFFSETS = {
    "train": 0,
    "val": 100_000,
    "test": 200_000,
}


def _make_route_straight(length: float, spacing: float = 2.0,
                         heading: float = 0.0) -> np.ndarray:
    """Generate a straight reference route as [R, 2]."""
    n = max(int(length / spacing), 2)
    t = np.linspace(0, length, n)
    xs = t * np.cos(heading)
    ys = t * np.sin(heading)
    return np.stack([xs, ys], axis=1)


def _make_route_curved(length: float, curvature: float,
                       spacing: float = 2.0) -> np.ndarray:
    """Generate a curved reference route as [R, 2]."""
    n = max(int(length / spacing), 2)
    if abs(curvature) < 1e-6:
        return _make_route_straight(length, spacing)
    radius = 1.0 / curvature
    arc_angle = length / radius
    angles = np.linspace(0, arc_angle, n)
    xs = radius * np.sin(angles)
    ys = radius * (1 - np.cos(angles))
    return np.stack([xs, ys], axis=1)


def _trajectory_from_route(route: np.ndarray, speed: float,
                           dt: float, horizon: int,
                           lateral_offset: float = 0.0,
                           speed_noise: float = 0.0,
                           rng: np.random.RandomState = None,
                           ) -> np.ndarray:
    """Build a [T, 4] trajectory (x, y, vx, vy) that follows a route.

    Interpolates along the route at the given speed, optionally adding
    lateral offset and per-step speed noise.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    # cumulative arc-length along route
    diffs = np.diff(route, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    waypoints = np.zeros((horizon, 4))
    s = 0.0  # distance along route
    for t in range(horizon):
        cur_speed = max(speed + (rng.randn() * speed_noise if speed_noise else 0.0), 0.0)
        s = min(s + cur_speed * dt, total_len - 1e-6)

        # interpolate position on route
        idx = np.searchsorted(cum_len, s, side="right") - 1
        idx = np.clip(idx, 0, len(route) - 2)
        frac = (s - cum_len[idx]) / max(seg_lens[idx], 1e-8)
        pos = route[idx] + frac * diffs[idx]

        # tangent direction for velocity and lateral offset
        tangent = diffs[idx] / max(seg_lens[idx], 1e-8)
        normal = np.array([-tangent[1], tangent[0]])

        waypoints[t, :2] = pos + lateral_offset * normal
        waypoints[t, 2:4] = tangent * cur_speed

    return waypoints


def _check_collided(candidate_wp: np.ndarray, agent_states: np.ndarray,
                    agent_mask: np.ndarray, dt: float,
                    vehicle_length: float = 4.5,
                    vehicle_width: float = 2.0) -> bool:
    """Check whether a trajectory actually collides with any agent.

    Uses simple circle overlap: collision when distance between ego and
    projected agent centre < (vehicle_length/2 + vehicle_width/2).
    This intentionally uses a simpler/tighter check than the continuous
    risk score so the binary flag captures *actual* overlap.
    """
    collision_radius = (vehicle_length + vehicle_width) / 2.0
    valid_agents = agent_states[agent_mask]
    if len(valid_agents) == 0:
        return False

    for t_idx, wp in enumerate(candidate_wp):
        ego_pos = wp[:2]
        future_time = t_idx * dt
        for agent in valid_agents:
            agent_pos = agent[:2] + agent[2:4] * future_time
            dist = np.linalg.norm(ego_pos - agent_pos)
            if dist < collision_radius:
                return True
    return False


class SyntheticScenarioGenerator:
    """Generates deterministic synthetic driving scenarios.

    Scenarios are fully reproducible given a seed. Each scenario contains
    multiple trajectory candidates with intentionally varied quality so
    that physics labels have good spread.

    Parameters
    ----------
    seed : int
        Base random seed.
    split : str
        One of "train", "val", "test". Uses disjoint seed ranges.
    num_scenarios : int
        Total scenarios to generate.
    num_candidates : int
        Trajectory candidates per scenario.
    horizon : int
        Trajectory length in timesteps.
    dt : float
        Timestep duration in seconds.
    """

    def __init__(
        self,
        seed: int = 42,
        split: str = "train",
        num_scenarios: int = 200,
        num_candidates: int = 8,
        horizon: int = 80,
        dt: float = 0.1,
    ):
        if split not in _SPLIT_OFFSETS:
            raise ValueError(f"split must be one of {list(_SPLIT_OFFSETS)}")
        self.base_seed = seed + _SPLIT_OFFSETS[split]
        self.split = split
        self.num_scenarios = num_scenarios
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.dt = dt

        self.physics_config = PhysicsConfig(dt=dt)
        self.physics_checker = PhysicsChecker(self.physics_config)

        # scenario type builders, cycled over
        self._builders = [
            self._build_straight_empty,
            self._build_straight_oncoming,
            self._build_curved,
            self._build_dense_traffic,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> List[SceneData]:
        """Generate all scenarios (deterministic given seed + split)."""
        scenes = []
        for i in range(self.num_scenarios):
            rng = np.random.RandomState(self.base_seed + i)
            builder = self._builders[i % len(self._builders)]
            scene = builder(rng, scene_idx=i)
            scenes.append(scene)
        return scenes

    def generate_with_labels(
        self,
    ) -> List[Tuple[SceneData, List[Dict[str, float]]]]:
        """Generate scenarios with physics pseudo-labels per candidate.

        Returns list of (scene, labels) where labels is a list (one dict
        per candidate) with keys: risk, comfort, progress, composite,
        collided (bool).
        """
        scenes = self.generate()
        results = []
        for scene in scenes:
            labels = []
            for cand in scene.candidates:
                scores = self.physics_checker.evaluate_trajectory(cand, scene)
                scores["collided"] = _check_collided(
                    cand.waypoints,
                    scene.agent_states,
                    scene.agent_mask,
                    self.dt,
                    self.physics_config.vehicle_length,
                    self.physics_config.vehicle_width,
                )
                labels.append(scores)
            results.append((scene, labels))
        return results

    # ------------------------------------------------------------------
    # Scenario builders
    # ------------------------------------------------------------------

    def _make_ego_state(self, x, y, vx, vy, heading=0.0) -> np.ndarray:
        """Create an [8] ego state: x, y, vx, vy, ax, ay, heading, yaw_rate."""
        return np.array([x, y, vx, vy, 0.0, 0.0, heading, 0.0], dtype=np.float64)

    def _make_agent(self, x, y, vx, vy, heading=0.0) -> np.ndarray:
        return np.array([x, y, vx, vy, 0.0, 0.0, heading, 0.0], dtype=np.float64)

    def _pad_agents(self, agents: List[np.ndarray], max_agents: int = 32):
        """Return (agent_states [N,8], agent_mask [N])."""
        n = len(agents)
        states = np.zeros((max_agents, 8), dtype=np.float64)
        mask = np.zeros(max_agents, dtype=bool)
        for i, a in enumerate(agents[:max_agents]):
            states[i] = a
            mask[i] = True
        return states, mask

    # -- 1. Straight road, no agents ----------------------------------

    def _build_straight_empty(self, rng: np.random.RandomState,
                              scene_idx: int) -> SceneData:
        speed = 8.0 + rng.rand() * 7.0  # 8-15 m/s
        route = _make_route_straight(length=120.0)
        ego = self._make_ego_state(0, 0, speed, 0)
        agents, mask = self._pad_agents([])

        candidates = []
        # good: follows route at speed
        candidates.append(self._cand(route, speed, rng, label="on_route"))
        # bad progress: stopped
        candidates.append(self._cand(route, 0.2, rng, label="stopped"))
        # bad progress: drifts off route
        candidates.append(self._cand(route, speed, rng, lateral=5.0, label="drift"))
        # slight drift
        candidates.append(self._cand(route, speed, rng, lateral=2.0, label="slight_drift"))

        # fill remaining with random variations
        while len(candidates) < self.num_candidates:
            lat = rng.uniform(-3, 3)
            sp = speed * rng.uniform(0.3, 1.2)
            candidates.append(self._cand(route, sp, rng, lateral=lat))

        return SceneData(
            ego_state=ego, lane_graph={},
            agent_states=agents, agent_mask=mask,
            route_waypoints=route,
            candidates=candidates[:self.num_candidates],
            scene_id=f"synth_{self.split}_{scene_idx:04d}_straight_empty",
            timestamp=0.0,
        )

    # -- 2. Straight road, oncoming agent -----------------------------

    def _build_straight_oncoming(self, rng: np.random.RandomState,
                                 scene_idx: int) -> SceneData:
        speed = 8.0 + rng.rand() * 7.0
        route = _make_route_straight(length=120.0)
        ego = self._make_ego_state(0, 0, speed, 0)

        # oncoming agent 40-60m ahead, approaching
        agent_x = 40.0 + rng.rand() * 20.0
        agent_speed = -(5.0 + rng.rand() * 10.0)
        agent_lateral = rng.uniform(-1.0, 1.0)
        agent = self._make_agent(agent_x, agent_lateral, agent_speed, 0, heading=np.pi)
        agents, mask = self._pad_agents([agent])

        candidates = []
        # safe: stay in lane
        candidates.append(self._cand(route, speed, rng, lateral=0.0, label="safe"))
        # risky: drift toward agent
        candidates.append(self._cand(route, speed, rng, lateral=agent_lateral, label="risky"))
        # evasive: swerve away
        swerve_dir = 3.0 if agent_lateral <= 0 else -3.0
        candidates.append(self._cand(route, speed * 0.8, rng, lateral=swerve_dir, label="evasive"))
        # very risky: head-on
        candidates.append(self._cand(route, speed * 1.2, rng, lateral=agent_lateral * 0.5,
                                     label="head_on"))

        while len(candidates) < self.num_candidates:
            lat = rng.uniform(-4, 4)
            sp = speed * rng.uniform(0.5, 1.2)
            candidates.append(self._cand(route, sp, rng, lateral=lat))

        return SceneData(
            ego_state=ego, lane_graph={},
            agent_states=agents, agent_mask=mask,
            route_waypoints=route,
            candidates=candidates[:self.num_candidates],
            scene_id=f"synth_{self.split}_{scene_idx:04d}_straight_oncoming",
            timestamp=0.0,
        )

    # -- 3. Curved road -----------------------------------------------

    def _build_curved(self, rng: np.random.RandomState,
                      scene_idx: int) -> SceneData:
        speed = 6.0 + rng.rand() * 6.0
        curvature = rng.choice([-1, 1]) * (0.01 + rng.rand() * 0.03)
        route = _make_route_curved(length=100.0, curvature=curvature)
        ego = self._make_ego_state(0, 0, speed, 0)
        agents, mask = self._pad_agents([])

        candidates = []
        # smooth: follows curve
        candidates.append(self._cand(route, speed, rng, label="smooth"))
        # jerky: high speed noise
        candidates.append(self._cand(route, speed, rng, speed_noise=4.0, label="jerky"))
        # cuts corner
        candidates.append(self._cand(route, speed * 1.1, rng, lateral=-2.0, label="corner_cut"))
        # wide
        candidates.append(self._cand(route, speed * 0.9, rng, lateral=2.0, label="wide"))

        while len(candidates) < self.num_candidates:
            lat = rng.uniform(-3, 3)
            sp = speed * rng.uniform(0.5, 1.3)
            sn = rng.uniform(0, 3.0)
            candidates.append(self._cand(route, sp, rng, lateral=lat, speed_noise=sn))

        return SceneData(
            ego_state=ego, lane_graph={},
            agent_states=agents, agent_mask=mask,
            route_waypoints=route,
            candidates=candidates[:self.num_candidates],
            scene_id=f"synth_{self.split}_{scene_idx:04d}_curved",
            timestamp=0.0,
        )

    # -- 4. Dense traffic ----------------------------------------------

    def _build_dense_traffic(self, rng: np.random.RandomState,
                             scene_idx: int) -> SceneData:
        speed = 5.0 + rng.rand() * 5.0
        route = _make_route_straight(length=120.0)
        ego = self._make_ego_state(0, 0, speed, 0)

        # 3-6 agents scattered ahead
        n_agents = rng.randint(3, 7)
        agent_list = []
        for _ in range(n_agents):
            ax = rng.uniform(15, 80)
            ay = rng.uniform(-4, 4)
            avx = rng.uniform(-3, 3)
            avy = rng.uniform(-1, 1)
            agent_list.append(self._make_agent(ax, ay, avx, avy))
        agents, mask = self._pad_agents(agent_list)

        candidates = []
        # conservative: slow, stay in lane
        candidates.append(self._cand(route, speed * 0.5, rng, label="conservative"))
        # aggressive: fast, straight through
        candidates.append(self._cand(route, speed * 1.3, rng, label="aggressive"))
        # weaving
        candidates.append(self._cand(route, speed, rng, lateral=3.0, label="weave_left"))
        candidates.append(self._cand(route, speed, rng, lateral=-3.0, label="weave_right"))

        while len(candidates) < self.num_candidates:
            lat = rng.uniform(-5, 5)
            sp = speed * rng.uniform(0.3, 1.4)
            candidates.append(self._cand(route, sp, rng, lateral=lat))

        return SceneData(
            ego_state=ego, lane_graph={},
            agent_states=agents, agent_mask=mask,
            route_waypoints=route,
            candidates=candidates[:self.num_candidates],
            scene_id=f"synth_{self.split}_{scene_idx:04d}_dense_traffic",
            timestamp=0.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cand(self, route, speed, rng, lateral=0.0, speed_noise=0.0,
              label="") -> TrajectoryCandidate:
        wp = _trajectory_from_route(
            route, speed, self.dt, self.horizon,
            lateral_offset=lateral,
            speed_noise=speed_noise,
            rng=rng,
        )
        ts = np.arange(self.horizon) * self.dt
        return TrajectoryCandidate(
            waypoints=wp, timestamps=ts, metadata={"label": label},
        )
