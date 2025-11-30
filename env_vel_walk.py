import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import threelp


class ThreeLPVelWalkEnv(gym.Env):
    """
    Velocity-tracking env: 1 step = 1 closed-form stride with foot-placement offset action.

    Observation (10D folded):
      [r_pel(2), r_sw(2), v_pel(2), v_sw(2), v_cmd(2)]  (all lateral folded by leg_flag)

    Action: 2D foot placement offset in stance frame (delta_p), scaled from tanh to [-a_max, a_max].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        t_ds: float = 0.1,
        t_ss: float = 0.6,
        params: threelp.ThreeLPParams | None = None,
        a_max: tuple[float, float] = (0.15, 0.08),
        cmd_x_range=(0.2, 1.2),
        cmd_y_range=(-0.2, 0.2),
        max_steps: int = 200,
        w_vel: float = 5.0,
        w_step: float = 1.0,
        w_act: float = 0.5,
        w_smooth: float = 0.2,
        r_fall: float = 100.0,
        bounds=None,
        seed: int | None = None,
    ):
        super().__init__()
        self.t_ds = t_ds
        self.t_ss = t_ss
        self.params = params or threelp.ThreeLPParams.Adult()
        self.a_max = np.asarray(a_max, dtype=np.float64)
        self.cmd_x_range = cmd_x_range
        self.cmd_y_range = cmd_y_range
        self.max_steps = max_steps
        self.w_vel = w_vel
        self.w_step = w_step
        self.w_act = w_act
        self.w_smooth = w_smooth
        self.r_fall = r_fall

        self.bounds = bounds or {
            "rpx_min": -0.15,
            "rpx_max": 0.35,
            "rpy_max": 0.15,
            "rsy_max": 0.30,
            "state_max": 10.0,
        }

        self.sim = threelp.ThreeLPSim(t_ds=self.t_ds, t_ss=self.t_ss, params=self.params, recenter_on_reset=True)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.state_vec = None
        self.leg_flag = 1
        self.v_cmd = None
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.prev_step = None
        self.step_count = 0
        self.step_time = self.t_ds + self.t_ss

    # -------- Helpers --------
    def _fold(self, y):
        return self.leg_flag * y

    def _sample_command(self):
        vx = self.rng.uniform(*self.cmd_x_range)
        vy = self.rng.uniform(*self.cmd_y_range)
        return np.array([vx, vy], dtype=np.float64)

    def _extract_rel(self, state_vec: np.ndarray):
        X2 = state_vec[0:2]
        x1 = state_vec[2:4]
        X3 = state_vec[4:6]
        X2dot = state_vec[6:8]
        x1dot = state_vec[8:10]
        r_pel = np.array([x1[0] - X3[0], self._fold(x1[1] - X3[1])])
        r_sw = np.array([X2[0] - X3[0], self._fold(X2[1] - X3[1])])
        v_pel = np.array([x1dot[0], self._fold(x1dot[1])])
        v_sw = np.array([X2dot[0], self._fold(X2dot[1])])
        return r_pel, r_sw, v_pel, v_sw

    def _build_obs(self):
        r_pel, r_sw, v_pel, v_sw = self._extract_rel(self.state_vec)
        v_cmd_fold = np.array([self.v_cmd[0], self._fold(self.v_cmd[1])])
        obs = np.concatenate([r_pel, r_sw, v_pel, v_sw, v_cmd_fold], axis=0).astype(np.float32)
        return obs

    def _action_to_delta_p(self, action: np.ndarray):
        a = np.tanh(action)
        return self.a_max * a

    def _compute_step_delta(self, prev_state_vec: np.ndarray):
        # With recentering, the previous swing rel vector is the step displacement.
        _, r_sw, _, _ = self._extract_rel(prev_state_vec)
        return r_sw

    def _check_bounds(self, r_pel, r_sw, v_pel, v_sw):
        b = self.bounds
        if r_pel[0] < b["rpx_min"] or r_pel[0] > b["rpx_max"]:
            return True
        if abs(r_pel[1]) > b["rpy_max"]:
            return True
        if abs(r_sw[1]) > b["rsy_max"]:
            return True
        if np.max(np.abs(np.concatenate([r_pel, r_sw, v_pel, v_sw]))) > b["state_max"]:
            return True
        return False

    # -------- Gym API --------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.leg_flag = int(self.rng.choice([-1, 1]))
        state0 = threelp.ThreeLPState()
        state0.q = [0.0] * 12
        self.sim.reset(state0, self.leg_flag)
        self.state_vec = np.array(self.sim.get_state().q, dtype=np.float64)
        self.v_cmd = self._sample_command()
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.prev_step = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        delta_p = self._action_to_delta_p(action)

        prev_state = self.state_vec.copy()
        step_delta = self._compute_step_delta(prev_state)

        state_struct, info_sim = self.sim.step_foot(delta_p.tolist())
        self.state_vec = np.array(state_struct.q, dtype=np.float64)
        self.leg_flag = info_sim.get("support_sign", self.leg_flag)

        obs = self._build_obs()
        r_pel, r_sw, v_pel, v_sw = self._extract_rel(self.state_vec)
        v_cmd_fold = np.array([self.v_cmd[0], self._fold(self.v_cmd[1])])
        vel_err = v_pel - v_cmd_fold

        # Reward
        cost = (
            self.w_vel * float(np.dot(vel_err, vel_err))
            + self.w_step * float(np.dot(step_delta - self.prev_step, step_delta - self.prev_step))
            + self.w_act * float(np.dot(action, action))
            + self.w_smooth * float(np.dot(action - self.prev_action, action - self.prev_action))
        )
        reward = -cost

        # Termination
        fallen = self._check_bounds(r_pel, r_sw, v_pel, v_sw)
        terminated = fallen
        truncated = self.step_count + 1 >= self.max_steps
        if fallen:
            reward -= self.r_fall

        self.prev_action = action.copy()
        self.prev_step = step_delta.copy()
        self.step_count += 1

        info = {
            "vel_err": vel_err,
            "step_delta": step_delta,
            "fallen": fallen,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.sim = None
