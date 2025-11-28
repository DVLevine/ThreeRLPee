# env_3lp.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from three_lp_py import ThreeLPSimPy


class ThreeLPGotoGoalEnv(gym.Env):
    """
    3LP-based goal-reaching environment.

    State:   3LP state (positions+velocities) + goal-relative features + phase
    Action:  hip + ankle torques (or 3LP input parameters) per time step

    You will need to adapt:
      - self.state_dim
      - self.action_dim
      - how state is split into (pelvis, feet, velocities)
      - reward shaping & termination conditions
      - actual calls into your C++ simulator
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.02,
        step_time: float = 0.4,
        max_episode_steps: int = 500,
        goal_radius: float = 0.1,
        max_action: float = 50.0,  # Nm or normalized units
        seed: int | None = None,
        use_python_sim: bool = True,
        sim_kwargs: dict | None = None,
    ):
        super().__init__()
        sim_kwargs = sim_kwargs or {}

        self.sim = ThreeLPSimPy(dt=dt, max_action=max_action, **sim_kwargs) if use_python_sim else None

        self.dt = self.sim.dt if self.sim is not None else dt
        self.step_time = self.sim.t_ds + self.sim.t_ss if self.sim is not None else step_time
        self.max_episode_steps = max_episode_steps
        self.goal_radius = goal_radius
        self.max_action = max_action

        # --- 3LP state dimensions ---
        # Q = [X2x, X2y, x1x, x1y, X3x, X3y, X2dx, X2dy, x1dx, x1dy, X3dx, X3dy]
        self.state_dim_3lp = self.sim.state_dim if self.sim is not None else 12

        # We'll append:
        #   goal_rel (2D)
        #   phase (1D)  ∈ [0, 1) within the current step
        self.goal_dim = 2
        self.phase_dim = 1

        self.obs_dim = self.state_dim_3lp + self.goal_dim + self.phase_dim

        # --- action space ---
        # For the Python simulator, we drive U (4 hip/ankle) + V (4 ramped torques).
        self.action_dim = self.sim.action_dim if self.sim is not None else 8

        self.action_space = spaces.Box(
            low=-self.max_action,
            high=self.max_action,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Deterministic goal sampling for reproducibility
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.state_3lp = None
        self.goal_world = None
        self.phase_time = 0.0
        self.sim_info = {}
        self.step_count = 0

    # ------------- Utility methods -------------

    def _sample_goal(self):
        """
        Sample a goal in world/ground frame.
        For now: random within a disk of radius R around origin.
        """
        r = self.np_random.uniform(0.5, 2.0)
        theta = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        goal = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
        return goal

    def _get_pelvis_pos(self, state_3lp: np.ndarray) -> np.ndarray:
        """
        Extract pelvis horizontal position from 3LP state.
        You must match this with your C++ 3LP state convention.

        Example if x = [X2x, X2y, x1x, x1y, X3x, X3y]:
        pelvis_xy = [x1x, x1y]
        """
        x = state_3lp[:6]  # positions only
        pelvis_xy = x[2:4]  # TODO: adjust indexing to your convention
        return pelvis_xy

    def _compute_observation(self) -> np.ndarray:
        pelvis_xy = self._get_pelvis_pos(self.state_3lp)
        goal_rel = self.goal_world - pelvis_xy  # in ground frame; optional: rotate into pelvis frame

        # Phase fraction within the current DS/SS segment
        phase_dur = self.sim_info.get("phase_duration", self.step_time) or self.step_time
        frac = float(self.sim_info.get("phase_time", self.phase_time)) / float(phase_dur) if phase_dur > 0 else 0.0
        frac = np.clip(frac, 0.0, 1.0)
        phase = np.array([frac], dtype=np.float32)

        obs = np.concatenate(
            [self.state_3lp.astype(np.float32), goal_rel.astype(np.float32), phase],
            axis=-1,
        )
        return obs

    def _compute_reward_and_done(self, obs: np.ndarray) -> tuple[float, bool, bool, dict]:
        # Distance to goal
        pelvis_xy = self._get_pelvis_pos(self.state_3lp)
        dist_to_goal = np.linalg.norm(self.goal_world - pelvis_xy)

        # Simple shaped reward: positive when moving closer to goal
        # Optionally keep previous distance in env state and use delta.
        reward = -dist_to_goal

        # Termination conditions
        reached = dist_to_goal < self.goal_radius
        time_limit = self.step_count >= self.max_episode_steps

        # Optional: fall detection (e.g. pelvis height < threshold)
        fallen = bool(self.sim_info.get("fallen", False)) if self.sim_info else False

        terminated = reached or fallen
        truncated = time_limit and not terminated

        info = {
            "dist_to_goal": dist_to_goal,
            "reached_goal": reached,
            "fallen": fallen,
        }

        return reward, terminated, truncated, info

    # ------------- Gym API -------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.step_count = 0
        self.phase_time = 0.0

        # Sample a new goal
        self.goal_world = self._sample_goal()

        if self.sim is not None:
            self.state_3lp = self.sim.reset().astype(np.float32)
            self.sim_info = {
                "phase": "ds",
                "phase_time": 0.0,
                "phase_duration": self.sim.phase_durations.get("ds", self.step_time),
                "support_sign": self.sim.support_sign,
                "fallen": False,
                "dt": self.sim.dt,
            }
        else:
            # Fallback placeholder
            self.state_3lp = np.zeros(self.state_dim_3lp, dtype=np.float32)
            self.sim_info = {"phase": "ds", "phase_time": 0.0, "phase_duration": self.step_time, "fallen": False}

        obs = self._compute_observation()
        info = {"goal_world": self.goal_world.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_action, self.max_action)

        if self.sim is not None:
            self.state_3lp, self.sim_info = self.sim.step(action)
            self.phase_time = float(self.sim_info.get("phase_time", self.phase_time + self.dt))
        else:
            # Placeholder if no simulator is wired
            self.state_3lp = self.state_3lp + 0.0 * action
            self.phase_time = (self.phase_time + self.dt) % self.step_time

        self.step_count += 1

        obs = self._compute_observation()
        reward, terminated, truncated, info = self._compute_reward_and_done(obs)
        info.update(self.sim_info)

        return obs, reward, terminated, truncated, info

    def render(self):
        # Optional: plot CoM, feet, goal, etc.
        pass

    def close(self):
        pass
