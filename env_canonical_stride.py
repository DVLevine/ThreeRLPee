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
                 reset_noise_std=0.01, failure_threshold=10.0, max_steps=200,
                 seed=None, debug_log=False, single_command_only=False,
                 zmp_limit: float = 0.25):
        super().__init__()
        self.params = params or threelp.ThreeLPParams.Adult()
        self.command_grid = np.array(commands, dtype=np.float64).reshape(-1)
        self.t_ds = t_ds
        self.t_ss = t_ss

        # Costs (defaults are light so survival bonus stays positive near reference).
        self.q_x = np.array(q_x_diag if q_x_diag is not None else [0.1] * 8, dtype=np.float64)
        self.r_u = np.array(r_u_diag if r_u_diag is not None else [0.01] * 8, dtype=np.float64)
        self.q_v = q_v

        self.u_limit = u_limit
        self.reset_noise_std = reset_noise_std
        self.failure_threshold = failure_threshold
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.maps = []
        for cmd in self.command_grid:
            res = threelp.build_canonical_stride(float(cmd), self.t_ds, self.t_ss, self.params)

            def _field(obj, name):
                if hasattr(obj, name):
                    return getattr(obj, name)
                try:
                    return obj[name]
                except Exception:
                    raise AttributeError(f"CanonicalStrideResult missing field {name}")

            cache = StrideCommandCache(
                command=float(cmd),
                A=np.array(_field(res, "A")),
                B=np.array(_field(res, "B")),
                b=np.array(_field(res, "b")),
                x_ref=np.array(_field(res, "x_ref")),
                u_ref=np.array(_field(res, "u_ref")),
                t_stride=float(_field(res, "t_stride")),
                success=bool(_field(res, "success")),
            )
            if cache.A.shape != (8, 8) or cache.B.shape != (8, 8):
                raise ValueError(f"build_canonical_stride returned shapes {cache.A.shape} and {cache.B.shape}, expected (8,8)")
            if not cache.success:
                if debug_log:
                    print(f"[env] skipping command {cmd} due to unsuccessful gait solve")
                continue
            self.maps.append(cache)
        if not self.maps:
            raise RuntimeError("No successful canonical stride maps available for the requested commands.")

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), np.float32)
        self.action_space = gym.spaces.Box(-u_limit, u_limit, (8,), np.float32)
        self.current = None
        self.delta_x = None
        # If True, always use the first command (no resampling) to diagnose single-speed stability.
        self.single_command_only = bool(single_command_only)
        self.zmp_limit = zmp_limit
        self.step_count = 0

    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        # Single-command mode: force the first command to avoid reference switches.
        self.current = self.maps[0] if self.single_command_only else self.rng.choice(self.maps)
        self.delta_x = self.rng.normal(0, self.reset_noise_std, 8)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        x_abs = self.current.x_ref + self.delta_x

        a = np.clip(action, -self.u_limit, self.u_limit)
        # Clip total torques as well to avoid exceeding physical limits when u_ref is large.
        u_applied = np.clip(self.current.u_ref + a, -self.u_limit, self.u_limit)

        # --- ZMP Check ---
        # Use sagittal ankle torque in canonical ordering:
        # canonical u = [Uh_y, Ua_y, Vh_y, Va_y, Uh_x, Ua_x, Vh_x, Va_x]
        # Index 1 corresponds to Ua_y (sagittal ankle). Adjust here if your ordering differs.
        mg = (self.params.m1 + 2 * self.params.m2) * self.params.g
        cop_x = u_applied[1] / mg
        if self.zmp_limit is not None and self.zmp_limit > 0 and abs(cop_x) > self.zmp_limit:
            return self._get_obs(), 0.0, True, False, {"fail": "ZMP", "cop_x": cop_x}

        x_next_abs = self.current.A @ x_abs + self.current.B @ u_applied + self.current.b
        self.delta_x = x_next_abs - self.current.x_ref

        # --- Costs ---
        c_state = np.dot(self.delta_x * self.q_x, self.delta_x)
        c_act = np.dot(a * self.r_u, a)  # penalize residual effort
        # Pelvis forward velocity (canonical index 2 per spec: [p, p_dot, f, f_dot])
        v_pelvis_x = x_next_abs[2]
        c_vel = self.q_v * (v_pelvis_x - self.current.command) ** 2

        # --- REWARD SHAPING ---
        # Keep survival bonus dominant near the reference; penalties should be <1 when close to ref.
        total_cost = c_state + c_act + c_vel
        reward = 1.0 - total_cost

        # Terminate on Divergence
        fail = np.any(np.abs(self.delta_x) > self.failure_threshold)
        truncated = self.step_count >= self.max_steps

        info = {}
        if fail:
            info["fail"] = "state_diverged"
        return self._get_obs(), reward, fail, truncated, info

    def _get_obs(self):
        s = np.array([2, 2, 2, 2, 1, 1, 1, 1], dtype=np.float32)
        scaled_dx = self.delta_x * s
        return np.concatenate([scaled_dx, [self.current.command]]).astype(np.float32)
