import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import threelp  # preferred backend (full feature set)
except Exception:
    threelp = None  # type: ignore


@dataclass
class ReferenceStride:
    """Cached reference trajectory for a command speed."""

    v_cmd: float
    phi: np.ndarray  # (N,) normalized stride phase in [0,1)
    x_ref: np.ndarray  # (N, 8) reduced canonical state
    q_ref: np.ndarray  # (N, 12) canonical 3LP state
    p_ref: np.ndarray  # (8,) torque parameters in stride order
    stride_time: float
    t_ds: float
    t_ss: float


class ThreeLPHighRateEnv(gym.Env):
    """
    High-rate 3LP RL environment that exposes the 10-D command-conditioned state
    described in the "High-Rate 3LP-Based RL" spec. Actions are 8-D deltas on
    the U/V torque basis, applied every Δt with exponential smoothing.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        t_ds: float = 0.1,
        t_ss: float = 0.6,
        dt: float = 0.02,
        max_steps: int = 2000,
        action_clip: float = 100.0,
        alpha_p: float = 0.05,
        p_decay: float = 0.98,
        alive_bonus: float = 10.0,
        q_e_diag: Tuple[float, ...] = (20.0, 20.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0),
        q_v: float = 5.0,
        r_u: float = 0.01,
        # |s1x|, |s1y| thresholds; extended to include velocity bounds (sag, lat)
        fall_bounds: Tuple[float, float, float, float] = (1.0, 0.5, 10.0, 10.0),
        v_cmd_range: Tuple[float, float] = (0.6, 1.4),
        ref_substeps: int = 120,
        reset_noise_std: float = 0.0,
        reference_cache: Optional[Dict[float, ReferenceStride]] = None,
        reference_builder: Optional[Callable[[float], ReferenceStride]] = None,
        seed: Optional[int] = None,
        random_phase: bool = True,
        obs_clip: float = 1e3,
    ):
        super().__init__()
        self.t_ds = float(t_ds)
        self.t_ss = float(t_ss)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.action_clip = float(action_clip)
        self.alpha_p = float(alpha_p)
        self.p_decay = float(p_decay)
        self.alive_bonus = float(alive_bonus)
        self.q_e = np.diag(np.asarray(q_e_diag, dtype=np.float64))
        self.q_v = float(q_v)
        self.r_u = float(r_u)
        if len(fall_bounds) == 2:
            self.fall_bounds = (float(fall_bounds[0]), float(fall_bounds[1]), 100.0, 100.0)
        else:
            self.fall_bounds = tuple(float(v) for v in fall_bounds)
        self.v_cmd_range = tuple(float(v) for v in v_cmd_range)
        self.ref_substeps = int(ref_substeps)
        self.reset_noise_std = float(reset_noise_std)
        self.reference_cache: Dict[float, ReferenceStride] = reference_cache or {}
        self.reference_builder = reference_builder
        self.rng = np.random.default_rng(seed)
        self.random_phase = bool(random_phase)
        self.obs_clip = float(obs_clip)
        assert self.t_ds > 1e-6 and self.t_ss > 1e-6, "Phase durations must be positive."

        self.stride_time = self.t_ds + self.t_ss
        self.obs_dim = 10
        self.action_dim = 8

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_clip, high=self.action_clip, shape=(self.action_dim,), dtype=np.float32)

        if threelp is None:
            raise RuntimeError("threelp pybind module is required for ThreeLPHighRateEnv.")
        self.sim = self._build_sim()
        self.current_ref: Optional[ReferenceStride] = None
        self.state_q = np.zeros(12, dtype=np.float64)
        self.support_sign = 1
        self.t_phase = 0.0
        self.phase = "ds"
        self.t_stride = 0.0
        self.step_count = 0
        self.p_running = np.zeros(self.action_dim, dtype=np.float64)
        self.v_cmd = float(self.v_cmd_range[0])
        self.state_q = np.zeros(12, dtype=np.float64)
        self.x_can = np.zeros(8, dtype=np.float64)

    # ------------------------------------------------------------------ Backend
    def _build_sim(self):
        params = threelp.ThreeLPParams.Adult()
        return threelp.ThreeLPSim(self.t_ds, self.t_ss, params, True)  # recenter_on_reset=True

    # ---------------------------------------------------------------- Reference
    def _default_reference_builder(self, v_cmd: float) -> ReferenceStride:
        if threelp is None:
            raise RuntimeError("threelp module is required to build reference gaits automatically.")
        params = threelp.ThreeLPParams.Adult()
        res = threelp.sample_reference_stride(float(v_cmd), self.t_ds, self.t_ss, params, self.ref_substeps)
        if not res.get("success", False):
            raise RuntimeError("sample_reference_stride failed for v_cmd={v_cmd}")
        phi_arr = np.asarray(res["phi"], dtype=np.float64).reshape(-1)
        x_arr = np.asarray(res["x_can"], dtype=np.float64)
        q_arr = np.asarray(res["q_can"], dtype=np.float64)
        p_ref = np.asarray(res["u_ref_stride"], dtype=np.float64).reshape(-1)
        q0 = np.asarray(res["q_ref0"], dtype=np.float64).reshape(-1)
        return ReferenceStride(
            v_cmd=float(v_cmd),
            phi=phi_arr,
            x_ref=x_arr,
            q_ref=q_arr,
            p_ref=p_ref,
            stride_time=self.stride_time,
            t_ds=self.t_ds,
            t_ss=self.t_ss,
        )

    def _get_reference(self, v_cmd: float) -> ReferenceStride:
        # Exact match cache first; users can pre-fill with custom references.
        if v_cmd in self.reference_cache:
            return self.reference_cache[v_cmd]
        builder = self.reference_builder or self._default_reference_builder
        ref = builder(v_cmd)
        self.reference_cache[v_cmd] = ref
        return ref

    def _interp_ref_values(self, ref: ReferenceStride, phi: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate reduced and full reference states at phase phi."""
        phi = float(phi - math.floor(phi))
        phi_grid = ref.phi
        if phi_grid.size < 2:
            return ref.x_ref[0], ref.q_ref[0]

        ph = np.concatenate([phi_grid, [phi_grid[0] + 1.0]])
        xr = np.vstack([ref.x_ref, ref.x_ref[0]])
        qr = np.vstack([ref.q_ref, ref.q_ref[0]])

        x_interp = np.empty(8, dtype=np.float64)
        for i in range(8):
            x_interp[i] = np.interp(phi, ph, xr[:, i])
        q_interp = np.empty(12, dtype=np.float64)
        for i in range(12):
            q_interp[i] = np.interp(phi, ph, qr[:, i])
        return x_interp, q_interp

    def _interp_ref_state(self, ref: ReferenceStride, phi: float) -> np.ndarray:
        phi = float(phi - math.floor(phi))  # keep in [0,1)
        phi_grid = ref.phi
        if phi_grid.size < 2:
            return ref.x_ref[0]
        ph = np.concatenate([phi_grid, [phi_grid[0] + 1.0]])
        xr = np.vstack([ref.x_ref, ref.x_ref[0]])
        x_interp = np.empty(8, dtype=np.float64)
        for i in range(8):
            x_interp[i] = np.interp(phi, ph, xr[:, i])
        return x_interp

    def reference_state(self, phi: float) -> np.ndarray:
        """Public helper to fetch the reduced reference state at phase φ."""
        if self.current_ref is None:
            return np.zeros(8, dtype=np.float64)
        x, _ = self._interp_ref_values(self.current_ref, phi)
        return x

    def _torque_correction(self, action: np.ndarray, theta_phase: float) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        theta = float(np.clip(theta_phase, 0.0, 1.0))
        u = a[:4]
        v = a[4:8]
        return u + theta * v  # shape (4,)

    # ---------------------------------------------------------------- Gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.t_phase = 0.0
        self.t_stride = 0.0
        self.phase = "ds"
        self.support_sign = 1

        self.v_cmd = float(options.get("v_cmd")) if options and "v_cmd" in options else float(
            self.rng.uniform(*self.v_cmd_range)
        )
        self.current_ref = self._get_reference(self.v_cmd)
        self.p_running = self.current_ref.p_ref.copy()

        # Randomize phase within the stride to expose stabilizable states.
        init_phi = float(self.rng.uniform(0.0, 1.0)) if self.random_phase else 0.0
        self.t_stride = init_phi * self.stride_time
        if self.t_stride < self.t_ds:
            self.phase = "ds"
            self.t_phase = self.t_stride
        else:
            self.phase = "ss"
            self.t_phase = self.t_stride - self.t_ds

        # Interpolate reference at this phase.
        _, q0 = self._interp_ref_values(self.current_ref, init_phi)
        if self.reset_noise_std > 0.0 or (options and options.get("perturb", False)):
            q0 = q0 + self.rng.normal(0.0, self.reset_noise_std, size=q0.shape)
        st = threelp.ThreeLPState()
        st.q = [float(v) for v in q0]
        self.sim.reset(st, 1)
        self.state_q = np.asarray(self.sim.get_state().q, dtype=np.float64)
        self.x_can = np.asarray(threelp.canonicalize_reduced_state(self.state_q.tolist(), self.support_sign), dtype=np.float64)

        obs = self._build_obs(self.x_can, init_phi)
        info = {"v_cmd": self.v_cmd}
        return obs.astype(np.float32), info

    def _build_obs(self, x_can: np.ndarray, phi: float) -> np.ndarray:
        return np.concatenate([x_can, [phi, self.v_cmd]], axis=0)

    def step(self, action: np.ndarray):
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        # Clip per-step delta tightly; accumulated params are clipped separately.
        act = np.clip(act, -10.0, 10.0)

        # Compute obs/state before stepping for reward.
        phi = float(self.t_stride / self.stride_time) if self.stride_time > 0 else 0.0
        obs = self._build_obs(self.x_can, phi)
        x = self.x_can
        x_ref, _ = self._interp_ref_values(self.current_ref, phi) if self.current_ref is not None else (np.zeros(8), np.zeros(12))
        e = x - x_ref
        v_forward = x[4]
        e_v = v_forward - self.v_cmd
        phase_duration = self.t_ds if self.phase == "ds" else self.t_ss
        theta_phase = (self.t_phase / phase_duration) if phase_duration > 0 else 0.0
        tau_corr, _ = threelp.compute_uv_torque(act.tolist(), self.phase, float(self.t_phase), self.t_ds, self.t_ss)
        cost = float(e.T @ self.q_e @ e + self.q_v * (e_v ** 2) + self.r_u * float(np.dot(tau_corr, tau_corr)))
        # Alive bonus keeps per-step reward positive near the reference.
        reward = self.dt * (self.alive_bonus - cost)

        # Update torque parameters with smoothing and optional leak back to reference.
        p_target = self.p_running + self.alpha_p * act
        decay = np.clip(self.p_decay, 0.0, 1.0)
        self.p_running = np.clip(decay * p_target + (1.0 - decay) * self.current_ref.p_ref, -self.action_clip, self.action_clip)

        # Advance simulator by dt.
        state_struct, x_next, info = self.sim.step_dt_augmented(self.p_running.tolist(), float(self.dt))
        support_prev = self.support_sign
        self.state_q = np.asarray(state_struct.q, dtype=np.float64)
        self.x_can = np.asarray(x_next, dtype=np.float64)
        self.support_sign = int(info.get("support_sign", self.support_sign))
        self.phase = info.get("phase", self.phase)
        self.t_phase = float(info.get("phase_time", self.t_phase + self.dt))
        phi_stride = float(info.get("phi_stride", (self.t_stride + self.dt) / self.stride_time))
        theta_phase = float(info.get("theta_phase", theta_phase))

        if self.support_sign != support_prev:
            self.t_stride = 0.0
        else:
            self.t_stride = phi_stride * self.stride_time
        self.step_count += 1

        obs_next = self._build_obs(self.x_can, phi_stride if self.stride_time > 0 else 0.0)

        # Termination: check position and velocity bounds and terminate on fall.
        s1x, s1y = obs_next[0], obs_next[1]
        ds1x, ds1y = obs_next[4], obs_next[5]
        fallen_pos = abs(s1x) > self.fall_bounds[0] or abs(s1y) > self.fall_bounds[1]
        fallen_vel = abs(ds1x) > self.fall_bounds[2] or abs(ds1y) > self.fall_bounds[3]
        fallen = fallen_pos or fallen_vel
        terminated = fallen
        truncated = self.step_count >= self.max_steps

        # Guard against non-finite or huge observations; treat as fall/explosion.
        if not np.all(np.isfinite(obs_next)) or np.any(np.abs(obs_next) > self.obs_clip):
            obs_next = np.clip(np.nan_to_num(obs_next, nan=0.0, posinf=self.obs_clip, neginf=-self.obs_clip),
                               -self.obs_clip, self.obs_clip)
            truncated = True
            terminated = True
            fallen = True
            reward = -10.0
        else:
            reward = float(np.clip(reward, -10.0, 10.0))

        # Torques based on actual running parameters and reference (for diagnostics/plots).
        tau_total, _ = threelp.compute_uv_torque(self.p_running.tolist(), self.phase, theta_phase, self.t_ds, self.t_ss)
        tau_ref = None
        if self.current_ref is not None:
            tau_ref, _ = threelp.compute_uv_torque(self.current_ref.p_ref.tolist(), self.phase, theta_phase, self.t_ds, self.t_ss)

        info_out = {
            "support_sign": self.support_sign,
            "phase": self.phase,
            "phase_time": self.t_phase,
            "phase_duration": phase_duration,
            "fallen": fallen,
            "phi_stride": phi_stride,
            "theta_phase": theta_phase,
            "p_running": self.p_running.tolist(),
            "tau_corr": np.asarray(tau_corr, dtype=np.float64).tolist(),
            "tau_total": np.asarray(tau_total, dtype=np.float64).tolist(),
            "tau_ref": np.asarray(tau_ref, dtype=np.float64).tolist() if tau_ref is not None else None,
            "state_world": np.asarray(state_struct.q, dtype=np.float64).tolist(),
            "x_can": self.x_can.tolist(),
            "action_applied": act.tolist(),
        }
        return obs_next.astype(np.float32), reward, bool(terminated), bool(truncated), info_out

    def render(self):
        return None

    def close(self):
        self.sim = None
