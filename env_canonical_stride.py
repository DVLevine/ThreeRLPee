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
        # Command scheduling: optionally resample each stride to train transitions.
        self.resample_command_each_step = False

        # ZMP safety guard (approximate)
        self.foot_length = 0.22
        self.max_cop = 0.5 * self.foot_length
        self.total_mass = float(self.params.m1 + 2.0 * self.params.m2)
        self.gravity = float(self.params.g)
        # Index of sagittal ankle torque in u vector; per spec order:
        # [Uh_y, Ua_y, Vh_y, Va_y, Uh_x, Ua_x, Vh_x, Va_x]
        self.ankle_sagittal_idx = 1
        self.zmp_penalty = 100.0

    # --------- Setup helpers ---------
    def _build_command_cache(self):
        self.maps.clear()
        for cmd in self.command_grid:
            res = threelp.build_canonical_stride(float(cmd), self.t_ds, self.t_ss, self.params)
            A = np.array(res["A"], dtype=np.float64)
            B = np.array(res["B"], dtype=np.float64)
            if A.shape != (8, 8) or B.shape != (8, 8):
                raise ValueError(f"build_canonical_stride returned shapes {A.shape} and {B.shape}, expected (8,8)")
            cache = StrideCommandCache(
                command=float(cmd),
                A=A,
                B=B,
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

        # Reconstruct absolute state at stride start.
        x_full_k = self.current.x_ref + self.delta_x

        # Apply action (clipped) and stride map for current command.
        a = np.asarray(action, dtype=np.float64)
        a = np.clip(a, -self.u_limit, self.u_limit)
        u_applied = self.current.u_ref + a
        u_applied = np.clip(u_applied, -self.u_limit, self.u_limit)
        delta_u = u_applied - self.current.u_ref

        x_full_k1 = self.current.A @ x_full_k + self.current.B @ u_applied + self.current.b

        # Optionally resample command each stride (for command-conditioned training).
        if self.resample_command_each_step and len(self.maps) > 1:
            self.current = self._sample_command()

        # Compute new error state against (possibly new) reference.
        delta_x_next = x_full_k1 - self.current.x_ref

        # ZMP/CoP safety guard using sagittal ankle torque.
        terminated = False
        fail_reason = None
        ankle_idx = self.ankle_sagittal_idx
        if 0 <= ankle_idx < u_applied.shape[0] and self.total_mass > 0 and self.gravity > 0:
            cop_x = float(u_applied[ankle_idx]) / (self.total_mass * self.gravity)
            if abs(cop_x) > self.max_cop:
                terminated = True
                fail_reason = "ZMP_violation"

        # Reward (negative cost)
        state_cost = float(np.dot(delta_x_next * self.q_x, delta_x_next))
        act_cost = float(np.dot(delta_u * self.r_u, delta_u))
        v_est = float(x_full_k1[2])  # pelvis forward velocity component
        speed_cost = self.q_v * (v_est - self.current.command) ** 2
        cost = state_cost + act_cost + speed_cost
        if fail_reason == "ZMP_violation":
            cost += self.zmp_penalty
        reward = -cost

        # Failure/termination checks
        if np.max(np.abs(delta_x_next)) > self.failure_threshold:
            terminated = True
            fail_reason = fail_reason or "state_diverged"
        truncated = self.step_count >= self.max_steps

        self.delta_x = delta_x_next
        obs = self._build_obs(self.delta_x, self.current.command)

        info = {
            "state_cost": state_cost,
            "act_cost": act_cost,
            "speed_cost": speed_cost,
            "v_est": v_est,
            "command": self.current.command,
        }
        if fail_reason:
            info["fail_reason"] = fail_reason
        if self.debug_log:
            info["delta_x_next_norm"] = float(np.linalg.norm(delta_x_next))
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.current = None
