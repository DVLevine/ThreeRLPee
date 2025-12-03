import math
from dataclasses import dataclass
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np

try:
    import threelp
except Exception:  # pragma: no cover - pybind may be unavailable in some environments
    threelp = None


@dataclass
class StrideCommandCache:
    command: float
    A: np.ndarray  # (8, 8)
    B: np.ndarray  # (8, 8)
    b: np.ndarray  # (8,)
    x_ref: np.ndarray  # (8,)
    u_ref: np.ndarray  # (8,)
    t_stride: float
    success: bool


class ThreeLPCanonicalStrideEnv(gym.Env):
    """
    Discrete stride-level 3LP environment in canonical coordinates (Phase A of the spec).

    Observation: [δx (8), command (1)]  — policy can append bias internally.
    Action:      δu (8) torque residuals on top of the reference stride coefficients.
    Transition:  δx_{k+1} = A δx_k + B δu_k   (in canonical space).
    Reward:      - (δx_{k+1}^T Q_x δx_{k+1} + δu^T R_u δu + q_v (v - v_des)^2).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        commands: Optional[Sequence[float]] = None,
        t_ds: float = 0.1,
        t_ss: float = 0.6,
        params: Optional["threelp.ThreeLPParams"] = None,
        q_x_diag: Optional[Sequence[float]] = None,
        r_u_diag: Optional[Sequence[float]] = None,
        q_v: float = 0.0,
        u_limit: float = 200.0,
        reset_noise_std: float = 0.01,
        failure_threshold: float = 5.0,
        max_steps: int = 200,
        seed: Optional[int] = None,
        debug_log: bool = False,
    ):
        super().__init__()
        if threelp is None:
            raise RuntimeError("threelp pybind module is required for canonical stride env")

        self.rng = np.random.default_rng(seed)
        self.t_ds = float(t_ds)
        self.t_ss = float(t_ss)
        self.params = params or threelp.ThreeLPParams.Adult()

        # Commands: list of forward speeds; Phase A uses a single constant command.
        self.command_grid = np.array(commands if commands is not None else [1.0], dtype=np.float64).reshape(-1)
        self.command_dim = 1

        # Penalties
        self.q_x = np.array(q_x_diag if q_x_diag is not None else [1.0] * 8, dtype=np.float64)
        self.r_u = np.array(r_u_diag if r_u_diag is not None else [0.1] * 8, dtype=np.float64)
        if self.q_x.shape != (8,) or self.r_u.shape != (8,):
            raise ValueError("q_x_diag and r_u_diag must be length-8 sequences")
        self.q_v = float(q_v)

        self.u_limit = float(u_limit)
        self.reset_noise_std = float(reset_noise_std)
        self.failure_threshold = float(failure_threshold)
        self.max_steps = int(max_steps)
        self.debug_log = debug_log

        self.maps: list[StrideCommandCache] = []
        self._build_command_cache()

        # Spaces
        obs_dim = 8 + self.command_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-self.u_limit, high=self.u_limit, shape=(8,), dtype=np.float32)

        # Episode state
        self.current: StrideCommandCache | None = None
        self.delta_x = np.zeros(8, dtype=np.float64)
        self.step_count = 0

    # --------- Setup helpers ---------
    def _build_command_cache(self):
        self.maps.clear()
        for cmd in self.command_grid:
            res = threelp.build_canonical_stride(float(cmd), self.t_ds, self.t_ss, self.params)
            cache = StrideCommandCache(
                command=float(cmd),
                A=np.array(res["A"], dtype=np.float64),
                B=np.array(res["B"], dtype=np.float64),
                b=np.array(res["b"], dtype=np.float64),
                x_ref=np.array(res["x_ref"], dtype=np.float64),
                u_ref=np.array(res["u_ref"], dtype=np.float64),
                t_stride=float(res.get("t_stride", self.t_ds + self.t_ss)),
                success=bool(res.get("success", True)),
            )
            self.maps.append(cache)
            if self.debug_log:
                print(f"[canonical_stride] cmd={cmd:.3f} success={cache.success} ||x_ref||={np.linalg.norm(cache.x_ref):.3f}")

    def _sample_command(self) -> StrideCommandCache:
        idx = self.rng.integers(0, len(self.maps))
        return self.maps[int(idx)]

    def _build_obs(self, delta_x: np.ndarray, cmd: float) -> np.ndarray:
        obs = np.concatenate([delta_x.astype(np.float32), np.array([cmd], dtype=np.float32)], axis=0)
        return obs

    # --------- Gym API ---------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0

        self.current = self._sample_command()
        noise = self.reset_noise_std * self.rng.standard_normal(8)
        x0 = self.current.x_ref + noise
        self.delta_x = x0 - self.current.x_ref

        obs = self._build_obs(self.delta_x, self.current.command)
        info = {"command": self.current.command, "success": self.current.success}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.current is not None, "Environment not reset"
        self.step_count += 1

        a = np.asarray(action, dtype=np.float64)
        a = np.clip(a, -self.u_limit, self.u_limit)
        u = self.current.u_ref + a
        u = np.clip(u, -self.u_limit, self.u_limit)
        delta_u = u - self.current.u_ref

        x_full = self.current.x_ref + self.delta_x
        x_next = self.current.A @ x_full + self.current.B @ u + self.current.b
        delta_x_next = x_next - self.current.x_ref

        # Reward (negative cost)
        state_cost = float(np.dot(delta_x_next * self.q_x, delta_x_next))
        act_cost = float(np.dot(delta_u * self.r_u, delta_u))
        # Approx pelvis forward speed from state element 2 (ṗ_x)
        v_est = float(x_next[2])
        speed_cost = self.q_v * (v_est - self.current.command) ** 2
        cost = state_cost + act_cost + speed_cost
        reward = -cost

        fail = np.max(np.abs(delta_x_next)) > self.failure_threshold
        truncated = self.step_count >= self.max_steps
        terminated = bool(fail)

        self.delta_x = delta_x_next
        obs = self._build_obs(self.delta_x, self.current.command)

        info = {
            "state_cost": state_cost,
            "act_cost": act_cost,
            "speed_cost": speed_cost,
            "v_est": v_est,
            "command": self.current.command,
        }
        if self.debug_log:
            info["delta_x_next_norm"] = float(np.linalg.norm(delta_x_next))
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.current = None
