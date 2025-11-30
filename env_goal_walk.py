import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import threelp  # pybind module built from ThreeLPee/src/three_lp_pybind.cpp


class ThreeLPGoalWalkEnv(gym.Env):
    """
    Goal-conditioned 3LP walking environment.

    One env step = one closed-form 3LP stride (DS->SS with stance swap).
    Action is a residual on nominal U/V torque basis coefficients, scaled from [-1,1]^8.
    Observation is a 14-D folded, normalized vector as per env_spec.md.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        goal_x_range=(1.0, 5.0),
        goal_y_range=(-1.0, 1.0),
        t_ds_base=0.1,
        t_ss_base=0.6,
        max_steps=200,
        # Nominal torques (per stance) for 1 m/s guess; order: [Uh_y, Uh_x, Ua_y, Ua_x]
        U_nom=np.array([20.0, 20.0, 100.0, 100.0], dtype=np.float64),
        V_nom=np.zeros(4, dtype=np.float64),
        alpha_U=0.5,
        alpha_V=0.5,
        floor_U=0.05,
        floor_V=0.05,
        reward_weights=None,
        noise_std=None,
        randomize_params=True,
        device="cpu",
    ):
        super().__init__()
        self.goal_x_range = goal_x_range
        self.goal_y_range = goal_y_range
        self.t_ds_base = t_ds_base
        self.t_ss_base = t_ss_base
        self.max_steps = max_steps
        self.U_nom_base = np.asarray(U_nom, dtype=np.float64)
        self.V_nom_base = np.asarray(V_nom, dtype=np.float64)
        self.alpha_U = alpha_U
        self.alpha_V = alpha_V
        self.floor_U = floor_U
        self.floor_V = floor_V
        self.randomize_params = randomize_params

        self.reward_weights = reward_weights or {
            "progress": 10.0,
            "alive": 0.5,
            "action": 0.01,
            "smooth": 0.001,
            "success": 20.0,
            "fail": -20.0,
        }

        self.noise_std = noise_std or {
            "pos": 0.01,
            "vel": 0.02,
            "goal": 0.01,
            "phase": 0.0,
        }

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.sim = None
        self.rng = np.random.default_rng()
        self.device = device

        self.L_ref = None
        self.V_ref = None
        self.prev_action = np.zeros(8, dtype=np.float64)
        self.d_goal_prev = None
        self.leg_flag = 1
        self.goal_world = None
        self.step_count = 0
        self.d_success = None
        self.pos_bound = None
        self.vel_bound = None

    # ---------- Helpers ----------

    def _sample_goal(self):
        gx = self.rng.uniform(*self.goal_x_range)
        gy = self.rng.uniform(*self.goal_y_range)
        return np.array([gx, gy], dtype=np.float64)

    def _randomize_params(self):
        params = threelp.ThreeLPParams.Adult()
        mass_scale = self.rng.uniform(0.95, 1.05)
        leg_scale = self.rng.uniform(0.98, 1.02)
        t_scale = self.rng.uniform(0.95, 1.05)

        params.m1 *= mass_scale
        params.m2 *= mass_scale
        params.h1 *= leg_scale
        params.h2 *= leg_scale
        params.h3 *= leg_scale
        params.wP *= leg_scale

        t_ds = self.t_ds_base * t_scale
        t_ss = self.t_ss_base * t_scale
        return params, t_ds, t_ss

    def _set_scales(self, params, t_ds, t_ss):
        # Reference length/velocity
        self.L_ref = params.h1
        self.V_ref = self.L_ref / (t_ds + t_ss)
        self.d_success = 0.1 * self.L_ref
        self.pos_bound = 3.0 * self.L_ref
        self.vel_bound = 10.0 * self.V_ref

    def _fold(self, y):
        return self.leg_flag * y

    def _build_obs(self, state_vec):
        # state_vec: np.ndarray (12,) [X2, x1, X3, X2dot, x1dot, X3dot]
        X2 = state_vec[0:2]
        x1 = state_vec[2:4]
        X3 = state_vec[4:6]
        X2dot = state_vec[6:8]
        x1dot = state_vec[8:10]

        r_pelvis = np.array([x1[0] - X3[0], self._fold(x1[1] - X3[1])])
        r_swing = np.array([X2[0] - X3[0], self._fold(X2[1] - X3[1])])
        v_pelvis = np.array([x1dot[0], self._fold(x1dot[1])])
        v_swing = np.array([X2dot[0], self._fold(X2dot[1])])

        g_rel = np.array([self.goal_world[0] - x1[0], self._fold(self.goal_world[1] - x1[1])])
        g_dist = np.linalg.norm(g_rel) + 1e-8
        g_heading = math.atan2(g_rel[1], g_rel[0])

        # Normalize
        r_pelvis_n = r_pelvis / self.L_ref
        r_swing_n = r_swing / self.L_ref
        v_pelvis_n = v_pelvis / self.V_ref
        v_swing_n = v_swing / self.V_ref
        g_rel_n = g_rel / self.L_ref
        g_dist_n = g_dist / self.L_ref
        g_heading_n = g_heading / math.pi  # (-1,1]
        phase_norm = 0.0  # step-based interface

        obs = np.array(
            [
                r_pelvis_n[0],
                r_pelvis_n[1],
                r_swing_n[0],
                r_swing_n[1],
                v_pelvis_n[0],
                v_pelvis_n[1],
                v_swing_n[0],
                v_swing_n[1],
                g_rel_n[0],
                g_rel_n[1],
                g_dist_n,
                g_heading_n,
                phase_norm,
                float(self.leg_flag),
            ],
            dtype=np.float32,
        )

        # Add noise on normalized components (except leg_flag)
        noise = np.zeros_like(obs)
        noise[:4] = self.rng.normal(0.0, self.noise_std["pos"], size=4)
        noise[4:8] = self.rng.normal(0.0, self.noise_std["vel"], size=4)
        noise[8:10] = self.rng.normal(0.0, self.noise_std["goal"], size=2)
        noise[11] += self.rng.normal(0.0, self.noise_std.get("heading", self.noise_std["goal"]))
        noise[12] += self.rng.normal(0.0, self.noise_std.get("phase", 0.0))
        obs_noisy = obs + noise
        obs_noisy[-1] = obs[-1]  # keep leg_flag exact
        return obs_noisy

    def _scale_action(self, action):
        a = np.tanh(action)  # ensure [-1,1]
        scale_U = self.alpha_U * (np.abs(self.U_nom) + self.floor_U)
        scale_V = self.alpha_V * (np.abs(self.V_nom) + self.floor_V)
        dU = scale_U * a[:4]
        dV = scale_V * a[4:]
        U = self.U_nom + dU
        V = self.V_nom + dV
        return np.concatenate([U, V], axis=0)

    # ---------- Gym API ----------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        params, t_ds, t_ss = self._randomize_params() if self.randomize_params else (threelp.ThreeLPParams.Adult(), self.t_ds_base, self.t_ss_base)
        self._set_scales(params, t_ds, t_ss)

        # Nominal torques per episode
        self.U_nom = self.U_nom_base.copy()
        self.V_nom = self.V_nom_base.copy()

        self.sim = threelp.ThreeLPSim(t_ds=t_ds, t_ss=t_ss, params=params, recenter_on_reset=True)

        self.leg_flag = int(self.rng.choice([-1, 1]))
        state0 = threelp.ThreeLPState()
        state0.q = [0.0] * 12
        state0.q[2] = params.wP  # small offset pelvis x?
        self.sim.reset(state0, self.leg_flag)

        self.goal_world = self._sample_goal()
        self.prev_action = np.zeros(8, dtype=np.float64)
        self.step_count = 0

        state_vec = np.array(self.sim.get_state().q, dtype=np.float64)
        obs = self._build_obs(state_vec)
        self.d_goal_prev = float(np.linalg.norm(self.goal_world - state_vec[2:4]))

        info = {"goal": self.goal_world.copy()}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        scaled = self._scale_action(action)

        # stride
        state_struct, info_sim = self.sim.step_closed_form(scaled.tolist())
        state_vec = np.array(state_struct.q, dtype=np.float64)
        self.leg_flag = info_sim["support_sign"]

        obs = self._build_obs(state_vec)

        d_goal = float(np.linalg.norm(self.goal_world - state_vec[2:4]))
        progress = self.d_goal_prev - d_goal
        self.d_goal_prev = d_goal

        # Reward
        w = self.reward_weights
        alive = 1.0
        r_progress = w["progress"] * progress
        r_alive = w["alive"] * alive
        r_action = -w["action"] * float(np.sum(action * action))
        delta_a = action - self.prev_action
        r_smooth = -w["smooth"] * float(np.sum(delta_a * delta_a))
        reward = r_progress + r_alive + r_action + r_smooth

        # Termination
        success = d_goal < self.d_success
        # Simple fall detection: blow-up of rel positions/velocities
        X2 = state_vec[0:2]; x1 = state_vec[2:4]; X3 = state_vec[4:6]
        X2dot = state_vec[6:8]; x1dot = state_vec[8:10]
        s1 = x1 - X3
        s2 = X2 - X3
        vel_norm = max(np.linalg.norm(X2dot), np.linalg.norm(x1dot))
        pos_norm = max(np.linalg.norm(s1), np.linalg.norm(s2))
        failure = bool(np.isnan(state_vec).any() or np.isinf(state_vec).any() or pos_norm > self.pos_bound or vel_norm > self.vel_bound)
        terminated = success or failure
        truncated = self.step_count + 1 >= self.max_steps

        if terminated:
            reward += w["success"] if success else w["fail"]

        self.prev_action = action.copy()
        self.step_count += 1

        info = {
            "dist_to_goal": d_goal,
            "success": success,
            "failure": failure,
            "support_sign": self.leg_flag,
            "t_ds": self.sim.t_ds,
            "t_ss": self.sim.t_ss,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.sim = None
