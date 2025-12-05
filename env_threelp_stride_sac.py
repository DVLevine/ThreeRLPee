import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import threelp  # C++ 3LP backend
except Exception:
    threelp = None  # type: ignore


# ---------------------------------------------------------------------------
# Reference stride data
# ---------------------------------------------------------------------------

@dataclass
class ReferenceStride:
    """
    Cached periodic reference trajectory for a given command speed.

    This is essentially a slice of the 3LP "gait manifold" at one v_cmd:
    - phi: normalized phase in [0, 1)
    - x_ref: reduced canonical state (8D)
    - q_ref: full 3LP state in world coordinates (12D)
    - p_ref: 8D U/V torque basis parameters that realize the reference gait
    """
    v_cmd: float
    phi: np.ndarray        # (N,)
    x_ref: np.ndarray      # (N, 8)
    q_ref: np.ndarray      # (N, 12)
    p_ref: np.ndarray      # (8,)
    stride_time: float
    t_ds: float
    t_ss: float


# ---------------------------------------------------------------------------
#  Stride-level RL environment around the 3LP gait manifold
# ---------------------------------------------------------------------------

class ThreeLPStrideEnv(gym.Env):
    """
    3LP stride-level RL environment for manifold tracking.

    * Dynamics: 3LP model (torso + swing + stance) with fixed (t_ds, t_ss).
    * Reference: periodic symmetric gait for a command speed v_cmd.
    * One RL step == one full 3LP stride (double-support + single-support).
    * Action: 8D residual on the reference U/V torque basis parameters,
      held constant for the whole stride.
    * Observation: error from the reference manifold at the stride boundary,
      plus the command speed.

      obs = concat( e, [v_cmd] )  where
          e = x_can - x_ref(phi=0)   (8D error in canonical reduced coords)

    * Reward: stride-integrated LQR-like cost (state error + velocity error)
      + small action penalty, plus an alive bonus per second.

    This is intended for off-policy methods like SAC.

    Gymnasium API:
        reset() -> (obs, info)
        step(a) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        # 3LP timing
        t_ds: float = 0.1,
        t_ss: float = 0.6,
        # internal integration step for sim (not RL step)
        inner_dt: float = 0.005,
        # episode length in strides (RL steps)
        max_strides: int = 200,
        # action scaling: env expects actions in [-1, 1], which are scaled
        # to residuals on the 8D U/V torque basis
        action_scale: float = 40.0,
        # reward shaping
        alive_bonus: float = 5.0,                 # per *second* of simulated time
        q_e_diag: Tuple[float, ...] = (
            20.0, 20.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0
        ),                                       # state error weights
        q_v: float = 2.0,                        # forward-velocity error weight
        r_action: float = 1e-3,                  # action L2 penalty
        terminal_penalty: float = 50.0,          # reward on fall
        # fall detection in canonical coords (s1x, s1y, ds1x, ds1y)
        fall_bounds: Tuple[float, float, float, float] = (1.0, 0.5, 10.0, 10.0),
        # command-speed range for random v_cmd
        v_cmd_range: Tuple[float, float] = (0.6, 1.4),
        # randomization
        reset_noise_std: float = 0.0,            # noise on q at episode start
        # manifold/reference handling
        ref_substeps: Optional[int] = None,
        reference_cache: Optional[Dict[float, ReferenceStride]] = None,
        reference_builder: Optional[Callable[[float], ReferenceStride]] = None,
        random_phase: bool = False,              # currently unused (stride-aligned reset)
        # misc
        seed: Optional[int] = None,
        obs_clip: float = 1e3,
    ):
        super().__init__()

        if threelp is None:
            raise RuntimeError(
                "threelp pybind module is required for ThreeLPStrideEnv "
                "(pip-install or build your 3LP bindings first)."
            )

        # --- Timing ---------------------------------------------------------
        self.t_ds = float(t_ds)
        self.t_ss = float(t_ss)
        assert self.t_ds > 1e-6 and self.t_ss > 1e-6, "Phase durations must be positive."
        self.stride_time = self.t_ds + self.t_ss
        self.inner_dt = float(inner_dt)

        # --- Episode length in strides --------------------------------------
        self.max_strides = int(max_strides)

        # --- Action scaling & penalties -------------------------------------
        self.action_dim = 8
        self.action_scale = float(action_scale)
        self.r_action = float(r_action)

        # SAC-style: policy outputs in [-1, 1], env rescales to meaningful δp
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # --- State / observation definition ---------------------------------
        # We work in "error coordinates" relative to the reference manifold:
        #   e = x_can - x_ref(phi=0)  (8D)  +  v_cmd (1D)  -> 9D total.
        self.obs_dim = 9
        self.obs_clip = float(obs_clip)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # --- Reward weights --------------------------------------------------
        self.alive_bonus = float(alive_bonus)
        self.q_e = np.diag(np.asarray(q_e_diag, dtype=np.float64))
        assert self.q_e.shape == (8, 8)
        self.q_v = float(q_v)
        self.terminal_penalty = float(terminal_penalty)

        # --- Fall detection --------------------------------------------------
        if len(fall_bounds) != 4:
            raise ValueError("fall_bounds must be (|s1x|, |s1y|, |ds1x|, |ds1y|).")
        self.fall_bounds = tuple(float(b) for b in fall_bounds)

        # --- Command speed & reference storage ------------------------------
        self.v_cmd_range = (float(v_cmd_range[0]), float(v_cmd_range[1]))
        self.reset_noise_std = float(reset_noise_std)

        self.reference_cache: Dict[float, ReferenceStride] = reference_cache or {}
        self.reference_builder = reference_builder  # if None, we use default builder
        self.ref_substeps = int(ref_substeps) if ref_substeps is not None else None

        # --- Randomness ------------------------------------------------------
        self.rng = np.random.default_rng(seed)
        self.random_phase = bool(random_phase)

        # --- 3LP simulator ---------------------------------------------------
        self.sim = self._build_sim()

        # 3LP internal state
        self.current_ref: Optional[ReferenceStride] = None
        self.v_cmd: float = self.v_cmd_range[0]
        self.support_sign: int = 1
        self.state_q = np.zeros(12, dtype=np.float64)   # world state (3LP q)
        self.x_can = np.zeros(8, dtype=np.float64)      # canonical reduced state
        self.t_in_stride: float = 0.0                   # time since last DS start
        self.stride_index: int = 0
        # Optional: when True, step() logs sub-step world states for visualization.
        self.capture_trajectory: bool = False

    # ------------------------------------------------------------------ Backend

    def _build_sim(self):
        params = threelp.ThreeLPParams.Adult()
        # recenter_on_reset=True: matches the original code's convention
        return threelp.ThreeLPSim(self.t_ds, self.t_ss, params, True)

    # ---------------------------------------------------------------- Reference

    def _default_reference_builder(self, v_cmd: float) -> ReferenceStride:
        """
        Build a periodic symmetric reference gait for given v_cmd by querying
        the C++ 3LP library. This is our "manifold" at that command speed.
        """
        params = threelp.ThreeLPParams.Adult()

        # Choose a reasonably fine sampling for the reference; either user-provided
        # substeps or a default that roughly aligns with inner_dt.
        stride_time = self.stride_time
        substeps_from_dt = int(math.ceil(stride_time / max(self.inner_dt, 1e-8)))
        substeps = (
            int(self.ref_substeps)
            if hasattr(self, "ref_substeps") and self.ref_substeps is not None
            else max(120, max(2, substeps_from_dt))
        )

        res = threelp.sample_reference_stride(
            float(v_cmd),
            self.t_ds,
            self.t_ss,
            params,
            substeps,
        )
        if not res.get("success", False):
            raise RuntimeError(f"sample_reference_stride failed for v_cmd={v_cmd}")

        phi_arr = np.asarray(res["phi"], dtype=np.float64).reshape(-1)
        x_arr = np.asarray(res["x_can"], dtype=np.float64)
        q_arr = np.asarray(res["q_can"], dtype=np.float64)
        p_ref = np.asarray(res["u_ref_stride"], dtype=np.float64).reshape(-1)

        return ReferenceStride(
            v_cmd=float(v_cmd),
            phi=phi_arr,
            x_ref=x_arr,
            q_ref=q_arr,
            p_ref=p_ref,
            stride_time=stride_time,
            t_ds=self.t_ds,
            t_ss=self.t_ss,
        )

    def _get_reference(self, v_cmd: float) -> ReferenceStride:
        """
        Return reference stride object for this v_cmd, using cache if available.
        """
        v_key = float(v_cmd)
        if v_key in self.reference_cache:
            return self.reference_cache[v_key]
        builder = self.reference_builder or self._default_reference_builder
        ref = builder(v_key)
        self.reference_cache[v_key] = ref
        return ref

    @staticmethod
    def _interp_periodic(
        phi_grid: np.ndarray,
        values: np.ndarray,
        phi: float,
    ) -> np.ndarray:
        """
        Generic periodic interpolation helper over [0, 1).

        phi_grid: shape (N,)
        values: shape (N, D)
        returns: shape (D,)
        """
        phi = float(phi - math.floor(phi))  # wrap to [0, 1)
        if phi_grid.size < 2:
            return values[0].copy()

        # extend cyclically by repeating first sample at phi+1
        ph = np.concatenate([phi_grid, [phi_grid[0] + 1.0]])
        v_ext = np.vstack([values, values[0]])

        out = np.empty(values.shape[1], dtype=np.float64)
        for i in range(values.shape[1]):
            out[i] = np.interp(phi, ph, v_ext[:, i])
        return out

    def _interp_ref_values(
        self, ref: ReferenceStride, phi: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate both reduced and full reference states at phase phi in [0,1).
        """
        x_interp = self._interp_periodic(ref.phi, ref.x_ref, phi)
        q_interp = self._interp_periodic(ref.phi, ref.q_ref, phi)
        return x_interp, q_interp

    def reference_state(self, phi: float) -> np.ndarray:
        """
        Public helper to fetch the reduced reference state at normalized phase φ.
        """
        if self.current_ref is None:
            return np.zeros(8, dtype=np.float64)
        x, _ = self._interp_ref_values(self.current_ref, phi)
        return x

    # ---------------------------------------------------------------- Gym API: reset

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset to (approximately) the reference manifold at φ = 0, with optional
        small noise. One RL episode is a sequence of strides from this start.

        options can contain:
            - "v_cmd": set command speed for this episode (float).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.stride_index = 0

        # 1) Choose command speed
        if options is not None and "v_cmd" in options:
            self.v_cmd = float(options["v_cmd"])
        else:
            self.v_cmd = float(self.rng.uniform(*self.v_cmd_range))

        self.current_ref = self._get_reference(self.v_cmd)

        # 2) Initialize simulator at reference DS-start pose (phi=0)
        x_ref0, q_ref0 = self._interp_ref_values(self.current_ref, 0.0)
        st = threelp.ThreeLPState()
        st.q = [float(v) for v in q_ref0]
        # By convention, reference builder uses +1 as initial support sign
        self.support_sign = 1
        self.sim.reset(st, self.support_sign)

        # 3) Optional: small perturbation around the manifold at reset
        if self.reset_noise_std > 0.0:
            noise = self.rng.normal(
                0.0, self.reset_noise_std, size=np.asarray(st.q).shape
            )
            q_noisy = np.asarray(st.q, dtype=np.float64) + noise
            st2 = threelp.ThreeLPState()
            st2.q = [float(v) for v in q_noisy]
            self.sim.reset(st2, self.support_sign)
            self.state_q = q_noisy
        else:
            self.state_q = np.asarray(q_ref0, dtype=np.float64)

        # 4) Canonical reduced state at reset
        self.x_can = np.asarray(
            threelp.canonicalize_reduced_state(
                self.state_q.tolist(), self.support_sign
            ),
            dtype=np.float64,
        )
        self.t_in_stride = 0.0

        # Observation: error at φ = 0 plus command speed
        e0 = self.x_can - x_ref0
        obs = self._build_obs(e0)

        info = {
            "v_cmd": self.v_cmd,
            "stride_index": self.stride_index,
            "phi": 0.0,
        }
        return obs, info

    def _build_obs(self, e: np.ndarray) -> np.ndarray:
        """
        Pack observation vector from error and command speed.
        """
        assert e.shape == (8,)
        obs = np.concatenate([e, np.array([self.v_cmd], dtype=np.float64)], axis=0)
        # Guard against NaN / inf
        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=self.obs_clip,
            neginf=-self.obs_clip,
        )
        obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        return obs.astype(np.float32)

    # ---------------------------------------------------------------- Gym API: step

    def step(self, action: np.ndarray):
        """
        Advance the 3LP simulation by exactly one stride (t_ds + t_ss) using a
        stride-constant 8D residual on the reference U/V torque basis.

            p_stride = p_ref + action_scale * tanh(action_raw)

        Internally, we integrate using self.inner_dt and accumulate a
        continuous-time cost integral as the reward signal.
        """
        if self.current_ref is None:
            raise RuntimeError("Environment not reset before step().")

        # --- Sanitize and scale action --------------------------------------
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] != self.action_dim:
            raise ValueError(
                f"Expected action shape ({self.action_dim},), got {act.shape}"
            )

        # SAC usually outputs in [-1,1]; we still clip for safety
        act = np.clip(act, -1.0, 1.0)
        delta_p = self.action_scale * act  # 8D residual on torque parameters

        # Reference torque parameters for this command speed
        p_ref = self.current_ref.p_ref
        if p_ref.shape[0] != self.action_dim:
            raise RuntimeError("Reference p_ref has unexpected dimension.")

        # Torque parameter used for this entire stride
        p_stride = p_ref + delta_p
        p_stride = np.asarray(p_stride, dtype=np.float64)
        p_list = p_stride.tolist()

        # --- Integrate one full stride --------------------------------------
        # Optional trajectory capture for visualization.
        traj_states = [] if self.capture_trajectory else None
        if traj_states is not None:
            traj_states.append(np.asarray(self.state_q, dtype=np.float64))

        total_cost = 0.0
        total_time = 0.0
        fallen = False
        phi = 0.0  # normalized phase during integration (for info)

        t_remaining = self.stride_time
        while t_remaining > 1e-9:
            dt_step = min(self.inner_dt, t_remaining)

            state_struct, x_next, info = self.sim.step_dt_augmented(
                p_list, float(dt_step)
            )

            self.state_q = np.asarray(state_struct.q, dtype=np.float64)
            self.x_can = np.asarray(x_next, dtype=np.float64)
            self.support_sign = int(info.get("support_sign", self.support_sign))
            phi = float(info.get("phi_stride", (self.t_in_stride + dt_step) / self.stride_time))
            if traj_states is not None:
                traj_states.append(self.state_q.copy())

            # Compute instantaneous cost at this sub-step
            x_ref, _ = self._interp_ref_values(self.current_ref, phi)
            e = self.x_can - x_ref

            # Forward pelvis velocity (sagittal) is x_can[4] in canonical coords
            v_forward = float(self.x_can[4])
            e_v = v_forward - self.v_cmd

            # Quadratic LQR-like running cost
            state_cost = float(e.T @ self.q_e @ e)
            vel_cost = self.q_v * (e_v ** 2)
            act_cost = self.r_action * float(np.dot(delta_p, delta_p))

            inst_cost = state_cost + vel_cost + act_cost
            total_cost += inst_cost * dt_step
            total_time += dt_step
            self.t_in_stride += dt_step
            t_remaining -= dt_step

            # Fall detection in canonical coordinates (based on s1 and ds1)
            s1x, s1y = float(self.x_can[0]), float(self.x_can[1])
            ds1x, ds1y = float(self.x_can[4]), float(self.x_can[5])

            fallen_pos = (
                abs(s1x) > self.fall_bounds[0]
                or abs(s1y) > self.fall_bounds[1]
            )
            fallen_vel = (
                abs(ds1x) > self.fall_bounds[2]
                or abs(ds1y) > self.fall_bounds[3]
            )
            fallen = fallen_pos or fallen_vel
            if fallen:
                break

        # One logical stride is done from the RL perspective
        self.stride_index += 1
        # Wrap local stride clock back to 0 so that next step always starts at DS
        self.t_in_stride = 0.0

        # --- Compute reward --------------------------------------------------
        if total_time <= 0.0:
            # Degenerate case; treat as fall
            reward = -self.terminal_penalty
            fallen = True
        else:
            # Alive bonus is per *second*, so integrate over stride duration
            reward = self.alive_bonus * total_time - total_cost

        if fallen:
            # Overwrite with a clean terminal penalty if we fell
            reward = -self.terminal_penalty

        # --- Next observation at new stride boundary ------------------------
        # We are now at the start of the next stride; φ effectively 0 again.
        x_ref0, _ = self._interp_ref_values(self.current_ref, 0.0)
        e_next = self.x_can - x_ref0
        obs_next = self._build_obs(e_next)

        # Guard obs again against explosions
        if not np.all(np.isfinite(obs_next)) or np.any(np.abs(obs_next) > self.obs_clip):
            obs_next = np.clip(
                np.nan_to_num(obs_next, nan=0.0, posinf=self.obs_clip, neginf=-self.obs_clip),
                -self.obs_clip,
                self.obs_clip,
            )
            fallen = True
            reward = -self.terminal_penalty

        terminated = bool(fallen)
        truncated = bool(self.stride_index >= self.max_strides)

        info_out = {
            "v_cmd": self.v_cmd,
            "stride_index": self.stride_index,
            "fallen": fallen,
            "phi_end": phi,
            "p_stride": p_stride.tolist(),
            "p_ref": p_ref.tolist(),
            "delta_p": delta_p.tolist(),
            "x_can": self.x_can.tolist(),
            "state_world": self.state_q.tolist(),
            "total_time": total_time,
        }
        if traj_states is not None:
            info_out["trajectory"] = [state.tolist() for state in traj_states]

        return obs_next, float(reward), terminated, truncated, info_out

    # ----------------------------------------------------------------- Misc API

    def render(self):
        # No built-in rendering; hook this up to your own 3D viewer if desired.
        return None

    def close(self):
        self.sim = None
