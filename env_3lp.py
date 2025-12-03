# env_3lp.py
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from three_lp_py import ThreeLPSimPy

try:
    import threelp

    ThreeLPSimCpp = threelp.ThreeLPSim  # type: ignore
    ThreeLPStateCpp = threelp.ThreeLPState  # type: ignore
    ThreeLPParamsCpp = threelp.ThreeLPParams  # type: ignore
except Exception:
    ThreeLPSimCpp = None


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
        max_episode_steps: int = 600,
        goal_radius: float = 0.1,
        max_action: float = 200.0,  # Nm or normalized units
        seed: int | None = None,
        use_python_sim: bool = True,
        sim_kwargs: dict | None = None,
        fall_vel_threshold: float = 5.0,
        fall_pos_threshold: float = 5.0,
        success_vel_thresh: float = 0.15,
        success_hold_steps: int = 3,
        near_goal_scale: float = 3.0,
        reward_weights: dict | None = None,
        debug_log: bool = False,
    ):
        super().__init__()
        sim_kwargs = sim_kwargs or {}

        if use_python_sim:
            self.sim = ThreeLPSimPy(dt=dt, max_action=max_action, **sim_kwargs)
        else:
            if ThreeLPSimCpp is None:
                raise RuntimeError("three_lp_cpp module not available; cannot use pybind sim")
            t_ds_arg = sim_kwargs.get("t_ds", 0.1)
            t_ss_arg = sim_kwargs.get("t_ss", 0.3)
            params_arg = sim_kwargs.get("params", ThreeLPParamsCpp.Adult() if ThreeLPParamsCpp else None)
            recenter = sim_kwargs.get("recenter_on_reset", True)
            self.sim = ThreeLPSimCpp(t_ds=t_ds_arg, t_ss=t_ss_arg, params=params_arg, recenter_on_reset=recenter)

        self.dt = self.sim.dt if self.sim is not None else dt
        self.step_time = self.sim.t_ds + self.sim.t_ss if self.sim is not None else step_time
        self.max_episode_steps = max_episode_steps
        self.goal_radius = goal_radius
        self.max_action = max_action
        self.fall_vel_threshold = fall_vel_threshold
        self.fall_pos_threshold = fall_pos_threshold
        self.success_vel_thresh = success_vel_thresh
        self.success_hold_steps = success_hold_steps
        self.near_goal_scale = near_goal_scale
        self.reward_weights = reward_weights or {
            "progress": 3.0,  # lower a bit
            "alive": 0.5,
            "action": 0.001,
            "smooth": 0.0005,
            "vel": 1.0,
            "success": 10.0,
            "fail": -100.0,  # much more negative

        #    "progress": 5.0,
        #    "alive": 0.5,
        #    "action": 0.001,
        #    "smooth": 0.0005,
        #    "success": 10.0,
        #    "fail": -10.0,
        }

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
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_dist = None
        self.debug_log = debug_log
        self.origin_world = np.zeros(2, dtype=np.float32)  # stance-foot world origin
        self.support_sign = 1
        self.goal_hold_counter = 0

    # ------------- Utility methods -------------

    def _sample_goal(self):
        """
        Sample a goal in world/ground frame.
        For now: random within a disk of radius R around origin.
        """
        #r = self.np_random.uniform(0.5, 2.0)
        r = self.np_random.uniform(0.5, 1.2)
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
        pelvis_xy = self._get_pelvis_pos(self.state_3lp) + self.origin_world
        goal_rel = self.goal_world - pelvis_xy  # world/ground frame

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
        pelvis_xy = self._get_pelvis_pos(self.state_3lp) + self.origin_world
        dist_to_goal = np.linalg.norm(self.goal_world - pelvis_xy)
        progress = self.prev_dist - dist_to_goal if self.prev_dist is not None else 0.0
        self.prev_dist = dist_to_goal

        w = self.reward_weights
        action_pen = w["action"] * float(np.dot(self.prev_action, self.prev_action))
        speed_pelvis = np.linalg.norm(self.state_3lp[8:10])
        near_goal = dist_to_goal < (self.near_goal_scale * self.goal_radius)
        vel_pen = w["vel"] * (speed_pelvis ** 2) if near_goal else 0.0
        reward = w["progress"] * progress + w["alive"] - action_pen - vel_pen

        # Termination conditions
        if dist_to_goal < self.goal_radius and speed_pelvis < self.success_vel_thresh:
            self.goal_hold_counter += 1
        else:
            self.goal_hold_counter = 0
        reached = self.goal_hold_counter >= self.success_hold_steps
        time_limit = self.step_count >= self.max_episode_steps

        # Fall detection: proxy using position/velocity blow-up.
        fallen = bool(self.sim_info.get("fallen", False)) if self.sim_info else False
        if not fallen:
            pos_norm = np.linalg.norm(self.state_3lp[:6])
            vel_norm = np.linalg.norm(self.state_3lp[6:])
            if pos_norm > self.fall_pos_threshold or vel_norm > self.fall_vel_threshold:
                fallen = True

        terminated = reached or fallen
        truncated = time_limit and not terminated

        #if terminated:
        #    reward += w["success"] if reached else w["fail"]

        if fallen:
            # Falling is *always* a failure, even if you're close to the goal.
            reward -= w["progress"] * progress
            reward += w["fail"]
        elif reached:
            # Only add success bonus if you are upright when you reach the goal.
            reward += w["success"]
        elif time_limit and not reached:
            reward += w["fail"] * 0.8  # or something small

        info = {
            "dist_to_goal": dist_to_goal,
            "reached_goal": reached,
            "fallen": fallen,
            "speed_pelvis": speed_pelvis,
            "near_goal": near_goal,
            "hold_steps": self.goal_hold_counter,
        }
        if self.debug_log:
            info.update(
                {
                    "progress": progress,
                    "pos_norm": np.linalg.norm(self.state_3lp[:6]),
                    "vel_norm": np.linalg.norm(self.state_3lp[6:]),
                    "vel_pen": vel_pen,
                }
            )

        return reward, terminated, truncated, info

    def _compute_reward_and_done_actiontype(self, obs: np.ndarray, action: np.ndarray) -> tuple[float, bool, bool, dict]:
        # Distance to goal
        pelvis_xy = self._get_pelvis_pos(self.state_3lp) + self.origin_world
        dist_to_goal = np.linalg.norm(self.goal_world - pelvis_xy)
        progress = self.prev_dist - dist_to_goal if self.prev_dist is not None else 0.0
        self.prev_dist = dist_to_goal
        speed_pelvis = np.linalg.norm(self.state_3lp[8:10])
        near_goal = dist_to_goal < (self.near_goal_scale * self.goal_radius)

        w = self.reward_weights
        # Penalize *current* action magnitude
        action_mag_pen = w["action"] * float(np.dot(action, action))

        # Penalize changes in action (smoothness)
        if self.prev_action is not None:
            diff = action - self.prev_action
            smooth_pen = w["smooth"] * float(np.dot(diff, diff))
        else:
            smooth_pen = 0.0

        vel_pen = w["vel"] * (speed_pelvis ** 2) if near_goal else 0.0
        reward = w["progress"] * progress + w["alive"] - action_mag_pen - smooth_pen - vel_pen

        # Termination conditions
        if dist_to_goal < self.goal_radius and speed_pelvis < self.success_vel_thresh:
            self.goal_hold_counter += 1
        else:
            self.goal_hold_counter = 0
        reached = self.goal_hold_counter >= self.success_hold_steps
        time_limit = self.step_count >= self.max_episode_steps

        # Fall detection: proxy using position/velocity blow-up.
        fallen = bool(self.sim_info.get("fallen", False)) if self.sim_info else False
        if not fallen:
            pos_norm = np.linalg.norm(self.state_3lp[:6])
            vel_norm = np.linalg.norm(self.state_3lp[6:])
            if pos_norm > self.fall_pos_threshold or vel_norm > self.fall_vel_threshold:
                fallen = True

        terminated = reached or fallen
        truncated = time_limit and not terminated

        # if terminated:
        #    reward += w["success"] if reached else w["fail"]

        if fallen:
            # Falling is *always* a failure, even if you're close to the goal.
            reward -= w["progress"] * progress
            reward += w["fail"]
        elif reached:
            # Only add success bonus if you are upright when you reach the goal.
            reward += w["success"]
        elif time_limit and not reached:
            reward += w["fail"] * 0.8  # or something small

        info = {
            "dist_to_goal": dist_to_goal,
            "reached_goal": reached,
            "fallen": fallen,
            "speed_pelvis": speed_pelvis,
            "near_goal": near_goal,
            "hold_steps": self.goal_hold_counter,
        }
        if self.debug_log:
            info.update(
                {
                    "progress": progress,
                    "pos_norm": np.linalg.norm(self.state_3lp[:6]),
                    "vel_norm": np.linalg.norm(self.state_3lp[6:]),
                    "vel_pen": vel_pen,
                }
            )

        return reward, terminated, truncated, info
    # ------------- Gym API -------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.step_count = 0
        self.phase_time = 0.0
        self.goal_hold_counter = 0

        # Sample a new goal
        self.goal_world = self._sample_goal()
        self.origin_world = np.zeros(2, dtype=np.float32)

        if self.sim is not None:
            if ThreeLPSimCpp is not None and isinstance(self.sim, ThreeLPSimCpp):
                state0 = ThreeLPStateCpp()
                state0.q = [0.0] * 12
                self.sim.reset(state0, 1)
                support = self.sim.support_sign
                self.state_3lp = np.array(self.sim.get_state().q, dtype=np.float32)
                phase_dur = getattr(self.sim, "t_ds", self.step_time)
            else:
                self.state_3lp = self.sim.reset().astype(np.float32)
                support = getattr(self.sim, "support_sign", 1.0)
                phase_dur = getattr(self.sim, "t_ds", self.step_time / 2)
            self.sim_info = {
                "phase": "ds",
                "phase_time": 0.0,
                "phase_duration": phase_dur,
                "support_sign": support,
                "fallen": False,
                "dt": self.dt,
            }
            self.support_sign = support
        else:
            # Fallback placeholder
            self.state_3lp = np.zeros(self.state_dim_3lp, dtype=np.float32)
            self.sim_info = {"phase": "ds", "phase_time": 0.0, "phase_duration": self.step_time, "fallen": False}

        obs = self._compute_observation()
        info = {"goal_world": self.goal_world.copy()}
        pelvis_xy = self._get_pelvis_pos(self.state_3lp) + self.origin_world
        self.prev_dist = float(np.linalg.norm(self.goal_world - pelvis_xy))
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_action, self.max_action)
        support_prev = self.support_sign
        swing_world_prev = self.state_3lp[:2] + self.origin_world  # swing foot world position before step

        if self.sim is not None:
            if ThreeLPSimCpp is not None and isinstance(self.sim, ThreeLPSimCpp):
                state_struct, info = self.sim.step_dt(action.tolist(), float(self.dt))
                self.state_3lp = np.array(state_struct.q, dtype=np.float32)
                self.sim_info = {
                    "phase_time": info.get("phase_time", 0.0),
                    "phase_duration": info.get("phase_duration", self.step_time),
                    "support_sign": info.get("support_sign", 1.0),
                    "fallen": info.get("fallen", False),
                }
                self.support_sign = self.sim_info["support_sign"]
            else:
                self.state_3lp, self.sim_info = self.sim.step(action)
                self.phase_time = float(self.sim_info.get("phase_time", self.phase_time + self.dt))
                self.support_sign = float(self.sim_info.get("support_sign", support_prev))
        else:
            # Placeholder if no simulator is wired
            self.state_3lp = self.state_3lp + 0.0 * action
            self.phase_time = (self.phase_time + self.dt) % self.step_time
            self.support_sign = support_prev

        self.step_count += 1
        if self.support_sign != support_prev:
            # New stance foot is previous swing foot; shift origin to keep world frame continuous.
            self.origin_world = swing_world_prev

        obs = self._compute_observation()
        #reward, terminated, truncated, info = self._compute_reward_and_done(obs)
        reward, terminated, truncated, info = self._compute_reward_and_done_actiontype(obs, action) #new
        info.update(self.sim_info)
        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        # Optional: plot CoM, feet, goal, etc.
        pass

    def close(self):
        pass
