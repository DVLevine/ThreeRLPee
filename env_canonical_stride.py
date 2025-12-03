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
                 reset_noise_std=0.01, reset_noise_to_state: bool = True,
                 action_noise_std: float = 0.0,
                 failure_threshold=10.0, max_steps=200,
                 seed=None, debug_log=False, single_command_only=False,
                 zmp_limit: float = 0.25, enforce_fixed_point: bool = False,
                 fail_penalty: float = 50.0,
                 obs_scale=None,
                 normalize_actions: bool = True):
        super().__init__()
        self.params = params or threelp.ThreeLPParams.Adult()
        self.command_grid = np.array(commands, dtype=np.float64).reshape(-1)
        self.t_ds = t_ds
        self.t_ss = t_ss

        # Costs (defaults light enough that near-ref stays positive).
        self.q_x = np.array(q_x_diag if q_x_diag is not None else [0.01] * 8, dtype=np.float64)
        self.r_u = np.array(r_u_diag if r_u_diag is not None else [0.001] * 8, dtype=np.float64)
        self.q_v = q_v

        self.u_limit = u_limit
        # Separate knobs: whether to perturb initial state and whether to inject action noise.
        self.reset_noise_std = reset_noise_std
        self.reset_noise_to_state = bool(reset_noise_to_state)
        self.action_noise_std = float(action_noise_std)
        self.failure_threshold = failure_threshold
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.enforce_fixed_point = bool(enforce_fixed_point)
        self.normalize_actions = bool(normalize_actions)

        # Observation scaling to bring positions ~O(1) and velocities ~O(1).
        # Ordering: [pelvis xy, swing foot xy, pelvis vel xy, swing vel xy].
        default_obs_scale = np.array([100.0, 100.0, 100.0, 100.0,
                                      10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        self.obs_scale = np.array(obs_scale, dtype=np.float32) if obs_scale is not None else default_obs_scale

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

            if self.enforce_fixed_point:
                # Optional: recompute x_ref to satisfy x_ref = A x_ref + B u_ref + b.
                I = np.eye(8, dtype=np.float64)
                rhs = cache.B @ cache.u_ref + cache.b
                try:
                    cache.x_ref = np.linalg.solve(I - cache.A, rhs)
                except np.linalg.LinAlgError:
                    cache.x_ref = np.linalg.pinv(I - cache.A) @ rhs

            self.maps.append(cache)
        if not self.maps:
            raise RuntimeError("No successful canonical stride maps available for the requested commands.")

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), np.float32)
        # Policies act in a normalized box; env rescales to physical torques.
        if self.normalize_actions:
            self.action_space = gym.spaces.Box(-1.0, 1.0, (8,), np.float32)
        else:
            self.action_space = gym.spaces.Box(-u_limit, u_limit, (8,), np.float32)
        self.current = None
        self.delta_x = None
        # If True, always use the first command (no resampling) to diagnose single-speed stability.
        self.single_command_only = bool(single_command_only)
        self.zmp_limit = zmp_limit
        self.step_count = 0
        self.fail_penalty = float(fail_penalty)

    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        # Single-command mode: force the first command to avoid reference switches.
        self.current = self.maps[0] if self.single_command_only else self.rng.choice(self.maps)
        # Optionally perturb the initial state; disable to keep the reference untouched.
        if self.reset_noise_to_state and self.reset_noise_std > 0:
            self.delta_x = self.rng.normal(0, self.reset_noise_std, 8)
        else:
            self.delta_x = np.zeros(8, dtype=np.float64)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        x_abs = self.current.x_ref + self.delta_x

        a = np.asarray(action, dtype=np.float64)
        # Work in normalized action space to keep gradients well scaled.
        if self.normalize_actions:
            if self.action_noise_std > 0:
                a = a + self.rng.normal(0, self.action_noise_std, size=a.shape)
            a_norm = np.clip(a, -1.0, 1.0)
            a_phys = a_norm * self.u_limit
        else:
            if self.action_noise_std > 0:
                a = a + self.rng.normal(0, self.action_noise_std, size=a.shape)
            a_phys = np.clip(a, -self.u_limit, self.u_limit)
            a_norm = a_phys / self.u_limit  # for cost shaping below

        # Clip total torques as well to avoid exceeding physical limits when u_ref is large.
        u_applied = np.clip(self.current.u_ref + a_phys, -self.u_limit, self.u_limit)

        # --- ZMP Check ---
        # Use sagittal ankle torque in canonical ordering:
        # canonical u = [Uh_y, Ua_y, Vh_y, Va_y, Uh_x, Ua_x, Vh_x, Va_x]
        # Index 1 corresponds to Ua_y (sagittal ankle). Adjust here if your ordering differs.
        mg = (self.params.m1 + 2 * self.params.m2) * self.params.g
        cop_x = u_applied[1] / mg
        if self.zmp_limit is not None and self.zmp_limit > 0 and abs(cop_x) > self.zmp_limit:
            # Heavy terminal penalty so "dying now" is worse than enduring several poor steps.
            return self._get_obs(), -self.fail_penalty, True, False, {"fail": "ZMP", "cop_x": cop_x}

        x_next_abs = self.current.A @ x_abs + self.current.B @ u_applied + self.current.b
        self.delta_x = x_next_abs - self.current.x_ref

        # --- Costs ---
        c_state = np.dot(self.delta_x * self.q_x, self.delta_x)
        # Penalize effort in normalized space so gradients stay sane; r_u_diag acts as weight.
        c_act = np.dot(a_norm * self.r_u, a_norm)
        # Optional speed tracking: compare to reference velocity (not command) to avoid bias.
        c_vel = 0.0
        if self.q_v != 0.0:
            v_ref = self.current.x_ref[2]  # pelvis forward velocity at stride start in canonical frame
            v_pelvis_x = x_next_abs[2]
            c_vel = self.q_v * (v_pelvis_x - v_ref) ** 2

        # --- REWARD SHAPING ---
        # Keep survival bonus dominant near the reference; penalties should be <1 when close to ref.
        total_cost = c_state + c_act + c_vel
        reward = 1.0 - total_cost

        # Terminate on Divergence
        fail = np.any(np.abs(self.delta_x) > self.failure_threshold)
        truncated = self.step_count >= self.max_steps

        info = {}
        if fail:
            reward -= self.fail_penalty
            info["fail"] = "state_diverged"
        return self._get_obs(), reward, fail, truncated, info

    def _get_obs(self):
        scaled_dx = self.delta_x * self.obs_scale
        return np.concatenate([scaled_dx, [self.current.command]]).astype(np.float32)
