import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stride_utils import canonical_action_to_stride, lift_canonical_state

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
    p_ref: np.ndarray  # (8,) torque parameters in policy/stride order
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
        action_clip: float = 200.0,
        alpha_p: float = 0.2,
        q_e_diag: Tuple[float, ...] = (20.0, 20.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0),
        q_v: float = 5.0,
        r_u: float = 0.01,
        fall_bounds: Tuple[float, float] = (0.6, 0.35),  # |s1x|, |s1y| thresholds
        v_cmd_range: Tuple[float, float] = (0.6, 1.4),
        ref_substeps: int = 120,
        reference_cache: Optional[Dict[float, ReferenceStride]] = None,
        reference_builder: Optional[Callable[[float], ReferenceStride]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.t_ds = float(t_ds)
        self.t_ss = float(t_ss)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.action_clip = float(action_clip)
        self.alpha_p = float(alpha_p)
        self.q_e = np.diag(np.asarray(q_e_diag, dtype=np.float64))
        self.q_v = float(q_v)
        self.r_u = float(r_u)
        self.fall_bounds = tuple(float(v) for v in fall_bounds)
        self.v_cmd_range = tuple(float(v) for v in v_cmd_range)
        self.ref_substeps = int(ref_substeps)
        self.reference_cache: Dict[float, ReferenceStride] = reference_cache or {}
        self.reference_builder = reference_builder
        self.rng = np.random.default_rng(seed)

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

    # ------------------------------------------------------------------ Backend
    def _build_sim(self):
        params = threelp.ThreeLPParams.Adult()
        return threelp.ThreeLPSim(self.t_ds, self.t_ss, params, True)  # recenter_on_reset=True

    # ---------------------------------------------------------------- Reference
    def _default_reference_builder(self, v_cmd: float) -> ReferenceStride:
        if threelp is None:
            raise RuntimeError("threelp module is required to build reference gaits automatically.")
        params = threelp.ThreeLPParams.Adult()
        res = threelp.build_canonical_stride(float(v_cmd), self.t_ds, self.t_ss, params)
        u_ref_canonical = np.asarray(getattr(res, "u_ref"), dtype=np.float64).reshape(-1)
        if u_ref_canonical.size != 8:
            raise RuntimeError("build_canonical_stride returned unexpected u_ref size")
        p_ref = canonical_action_to_stride(u_ref_canonical)
        x0 = np.asarray(getattr(res, "x_ref"), dtype=np.float64).reshape(-1)
        if x0.size != 8:
            raise RuntimeError("build_canonical_stride returned unexpected x_ref size")
        q0 = lift_canonical_state(x0)

        # Sample a dense stride using step_dt to get time-aligned reduced reference.
        sim = threelp.ThreeLPSim(self.t_ds, self.t_ss, params, True)
        st = threelp.ThreeLPState()
        st.q = [float(v) for v in q0]
        sim.reset(st, 1)

        N = max(10, self.ref_substeps)
        phi_grid = []
        x_grid = []
        q_grid = []
        t_stride = 0.0
        dt_ref = self.stride_time / float(N)
        support = 1
        state_struct = sim.get_state()
        q_now = np.asarray(state_struct.q, dtype=np.float64)
        for _ in range(N):
            phi = (t_stride / self.stride_time) % 1.0
            q_can = self._canonicalize_state(q_now, support)
            x = self._project_to_reduced_state(q_can)
            phi_grid.append(phi)
            x_grid.append(x)
            q_grid.append(q_can.copy())

            state_struct, info = sim.step_dt(p_ref.tolist(), float(dt_ref))
            q_now = np.asarray(state_struct.q, dtype=np.float64)
            support = int(info.get("support_sign", support))
            t_stride += dt_ref
            if t_stride >= self.stride_time:
                break

        phi_arr = np.asarray(phi_grid, dtype=np.float64)
        order = np.argsort(phi_arr)
        phi_arr = phi_arr[order]
        x_arr = np.asarray(x_grid, dtype=np.float64)[order]
        q_arr = np.asarray(q_grid, dtype=np.float64)[order]
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

    # ---------------------------------------------------------------- Canonicalization / features
    def _canonicalize_state(self, q: np.ndarray, support_sign: int) -> np.ndarray:
        """
        Convert a world-frame 12D state into left-support canonical frame (12D).
        We avoid the pybind canonicalize_state because it returns the reduced 8D state.
        """
        q_can = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_can.shape[0] != 12:
            raise ValueError(f"Expected 12D state, got shape {q_can.shape}")
        if support_sign < 0:
            # Swap swing/stance blocks (pos and vel)
            q_can[0:2], q_can[4:6] = q_can[4:6].copy(), q_can[0:2].copy()
            q_can[6:8], q_can[10:12] = q_can[10:12].copy(), q_can[6:8].copy()
            # Mirror lateral components (y indices)
            q_can[[1, 3, 5, 7, 9, 11]] *= -1.0
        # Recenter to stance foot origin and zero stance velocity drift.
        stance_xy = q_can[4:6].copy()
        q_can[0:2] -= stance_xy
        q_can[2:4] -= stance_xy
        q_can[4:6] -= stance_xy
        stance_vel = q_can[10:12].copy()
        q_can[6:8] -= stance_vel
        q_can[8:10] -= stance_vel
        q_can[10:12] -= stance_vel
        return q_can

    def _project_to_reduced_state(self, q_can: np.ndarray) -> np.ndarray:
        q = np.asarray(q_can, dtype=np.float64).reshape(-1)
        s1 = q[2:4] - q[4:6]  # pelvis - stance
        s2 = q[0:2] - q[2:4]  # swing - pelvis
        ds1 = q[8:10] - q[10:12]
        ds2 = q[6:8] - q[8:10]
        return np.concatenate([s1, s2, ds1, ds2], axis=0)

    def _interp_ref_state(self, ref: ReferenceStride, phi: float) -> np.ndarray:
        phi = float(phi % 1.0)
        # Wrap phi grid for interpolation
        phi_grid = ref.phi
        if phi_grid.size < 2:
            return ref.x_ref[0]
        # Ensure last point >1 for wrap-around
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
        return self._interp_ref_state(self.current_ref, phi)

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

        # Initialize simulator at the reference fixed point.
        q0 = self.current_ref.q_ref[0] if self.current_ref.q_ref.size > 0 else lift_canonical_state(np.zeros(8))
        st = threelp.ThreeLPState()
        st.q = [float(v) for v in q0]
        self.sim.reset(st, 1)
        self.state_q = np.asarray(self.sim.get_state().q, dtype=np.float64)

        obs = self._build_obs(self.state_q, self.support_sign, self.t_stride)
        info = {"v_cmd": self.v_cmd}
        return obs.astype(np.float32), info

    def _build_obs(self, q_world: np.ndarray, support_sign: int, t_stride: float) -> np.ndarray:
        q_can = self._canonicalize_state(q_world, support_sign)
        x = self._project_to_reduced_state(q_can)
        phi = (t_stride % self.stride_time) / self.stride_time
        return np.concatenate([x, [phi, self.v_cmd]], axis=0)

    def step(self, action: np.ndarray):
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        act = np.clip(act, -self.action_clip, self.action_clip)

        # Compute obs/state before stepping for reward.
        obs = self._build_obs(self.state_q, self.support_sign, self.t_stride)
        x = obs[:8]
        phi = float(obs[8])
        x_ref = self._interp_ref_state(self.current_ref, phi) if self.current_ref is not None else np.zeros(8)
        e = x - x_ref
        v_forward = x[4]
        e_v = v_forward - self.v_cmd
        theta_phase = (self.t_phase / (self.t_ds if self.phase == "ds" else self.t_ss)) if (
            self.t_ds > 0 and self.t_ss > 0
        ) else 0.0
        tau_corr = self._torque_correction(act, theta_phase)
        cost = float(e.T @ self.q_e @ e + self.q_v * (e_v ** 2) + self.r_u * float(np.dot(tau_corr, tau_corr)))
        reward = -self.dt * cost

        # Update torque parameters with smoothing and clip.
        self.p_running = np.clip(self.p_running + self.alpha_p * act, -self.action_clip, self.action_clip)

        # Advance simulator by dt.
        state_struct, info = self.sim.step_dt(self.p_running.tolist(), float(self.dt))
        support_prev = self.support_sign
        self.state_q = np.asarray(state_struct.q, dtype=np.float64)
        self.support_sign = int(info.get("support_sign", self.support_sign))
        phase_duration = float(info.get("phase_duration", self.t_ds))
        self.phase = "ds" if math.isclose(phase_duration, self.t_ds, rel_tol=1e-4, abs_tol=1e-4) else "ss"
        self.t_phase = float(info.get("phase_time", self.t_phase + self.dt))

        if self.support_sign != support_prev:
            self.t_stride = 0.0
        else:
            self.t_stride += self.dt
            if self.t_stride >= self.stride_time:
                self.t_stride -= self.stride_time
        self.step_count += 1

        obs_next = self._build_obs(self.state_q, self.support_sign, self.t_stride)

        # Termination: pelvis relative displacement exceeds bounds or step limit.
        s1x, s1y = obs_next[0], obs_next[1]
        fallen = abs(s1x) > self.fall_bounds[0] or abs(s1y) > self.fall_bounds[1]
        terminated = fallen
        truncated = self.step_count >= self.max_steps
        if fallen:
            reward -= 200.0

        info_out = {
            "support_sign": self.support_sign,
            "phase": self.phase,
            "phase_time": self.t_phase,
            "phase_duration": phase_duration,
            "fallen": fallen,
        }
        return obs_next.astype(np.float32), float(reward), bool(terminated), bool(truncated), info_out

    def render(self):
        return None

    def close(self):
        self.sim = None
