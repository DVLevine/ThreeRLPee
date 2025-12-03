import gymnasium as gym
import numpy as np
from dataclasses import dataclass

try:
    import threelp
except Exception:
    threelp = None


@dataclass
class StrideCommandCache:
    command: float
    A: np.ndarray
    B: np.ndarray
    b: np.ndarray
    x_ref: np.ndarray
    u_ref: np.ndarray
    t_stride: float
    success: bool


class ThreeLPCanonicalStrideEnv(gym.Env):
    def __init__(self, commands=[1.0], t_ds=0.1, t_ss=0.6, params=None,
                 q_x_diag=None, r_u_diag=None, q_v=0.0, u_limit=200.0,
                 reset_noise_std=0.01, failure_threshold=5.0, max_steps=200,
                 seed=None, debug_log=False):
        super().__init__()
        self.params = params or threelp.ThreeLPParams.Adult()
        self.command_grid = np.array(commands, dtype=np.float64).reshape(-1)
        self.t_ds = t_ds
        self.t_ss = t_ss

        # Costs (reduced weights since we now use survival bonus)
        self.q_x = np.array(q_x_diag if q_x_diag else [1.0] * 8, dtype=np.float64)
        self.r_u = np.array(r_u_diag if r_u_diag else [0.1] * 8, dtype=np.float64)
        self.q_v = q_v

        self.u_limit = u_limit
        self.reset_noise_std = reset_noise_std
        self.failure_threshold = failure_threshold
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.maps = []
        for cmd in self.command_grid:
            res = threelp.build_canonical_stride(float(cmd), self.t_ds, self.t_ss, self.params)
            self.maps.append(StrideCommandCache(
                command=float(cmd),
                A=np.array(res["A"]), B=np.array(res["B"]), b=np.array(res["b"]),
                x_ref=np.array(res["x_ref"]), u_ref=np.array(res["u_ref"]),
                t_stride=res["t_stride"], success=res["success"]
            ))

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), np.float32)
        self.action_space = gym.spaces.Box(-u_limit, u_limit, (8,), np.float32)
        self.current = None
        self.delta_x = None

    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.current = self.rng.choice(self.maps)
        self.delta_x = self.rng.normal(0, self.reset_noise_std, 8)
        return self._get_obs(), {}

    def step(self, action):
        x_abs = self.current.x_ref + self.delta_x

        a = np.clip(action, -self.u_limit, self.u_limit)
        u_applied = self.current.u_ref + a

        # --- ZMP Check ---
        mg = (self.params.m1 + 2 * self.params.m2) * self.params.g
        cop_x = u_applied[2] / mg
        # ZMP Violation is a termination, but we don't apply massive negative penalty.
        # We just stop earning the survival bonus.
        if abs(cop_x) > 0.15:
            return self._get_obs(), 0.0, True, False, {"fail": "ZMP"}

        x_next_abs = self.current.A @ x_abs + self.current.B @ u_applied + self.current.b
        self.delta_x = x_next_abs - self.current.x_ref

        # --- Costs ---
        c_state = np.dot(self.delta_x * self.q_x, self.delta_x)
        c_act = np.dot(a * self.r_u, a)
        c_vel = self.q_v * (x_next_abs[2] - self.current.command) ** 2

        # --- REWARD SHAPING ---
        # 1. Survival Bonus: Primary driver of learning early on.
        # 2. Penalty: Secondary objective to clean up the gait.
        reward = 1.0 - 0.1 * (c_state + c_act + c_vel)

        # Terminate on Divergence
        fail = np.any(np.abs(self.delta_x) > self.failure_threshold)

        return self._get_obs(), reward, fail, False, {}

    def _get_obs(self):
        s = np.array([2, 2, 2, 2, 1, 1, 1, 1], dtype=np.float32)
        scaled_dx = self.delta_x * s
        return np.concatenate([scaled_dx, [self.current.command]]).astype(np.float32)