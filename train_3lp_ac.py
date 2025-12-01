# train_3lp_ac.py
import atexit
import multiprocessing as mp
import queue

import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from policy_3lp import (
    PolicyConfig,
    LinearBasisActor,
    QuadraticCritic,
    MLPActor,
    MLPCritic,
)


class RunningNorm:
    def __init__(self, epsilon=1e-8, shape=None, device="cpu"):
        self.mean = None
        self.var = None
        self.count = epsilon
        self.device = device
        if shape is not None:
            self.mean = torch.zeros(shape, device=device)
            self.var = torch.ones(shape, device=device)

    def update(self, x: torch.Tensor):
        if self.mean is None:
            self.mean = torch.zeros_like(x.mean(0))
            self.var = torch.ones_like(self.mean)
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor):
        if self.mean is None:
            return x
        return (x - self.mean) / torch.sqrt(self.var + 1e-6)


def collect_rollout(env, actor, critic, horizon=2048, gamma=0.99, lam=0.95, device="cpu", obs_norm: RunningNorm | None = None):
    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    val_buf = []
    done_buf = []
    trunc_buf = []

    obs, info = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs_norm:
        obs_norm.update(obs.unsqueeze(0))
        obs_n = obs_norm.normalize(obs.unsqueeze(0)).squeeze(0)
    else:
        obs_n = obs

    for t in range(horizon):
        with torch.no_grad():
            # Build basis and value
            if hasattr(actor, "encoder") and hasattr(actor.encoder, "critic_features"):
                critic_phi = actor.encoder.critic_features(obs_n.unsqueeze(0))
                value = critic(critic_phi).squeeze(0)
            else:
                value = critic(obs_n.unsqueeze(0)).squeeze(0)
            action, logp, _ = actor.act(obs_n.unsqueeze(0))
        action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        obs_buf.append(obs_n)
        act_buf.append(action.squeeze(0))
        logp_buf.append(logp.squeeze(0))
        rew_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
        val_buf.append(value)
        done_buf.append(torch.tensor(float(terminated), dtype=torch.float32, device=device))
        trunc_buf.append(torch.tensor(float(truncated), dtype=torch.float32, device=device))

        obs = next_obs_t
        if obs_norm:
            obs_norm.update(obs.unsqueeze(0))
            obs_n = obs_norm.normalize(obs.unsqueeze(0)).squeeze(0)
        else:
            obs_n = obs
        if terminated or truncated:
            obs, info = env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_norm:
                obs_norm.update(obs.unsqueeze(0))
                obs_n = obs_norm.normalize(obs.unsqueeze(0)).squeeze(0)
            else:
                obs_n = obs

    # Bootstrap value for the last observation
    with torch.no_grad():
        if hasattr(actor, "encoder") and hasattr(actor.encoder, "critic_features"):
            phi_last = actor.encoder.critic_features(obs_n.unsqueeze(0))
            next_value = critic(phi_last).squeeze(0)
        else:
            next_value = critic(obs_n.unsqueeze(0)).squeeze(0)

    # Convert to tensors
    obs_buf = torch.stack(obs_buf)
    act_buf = torch.stack(act_buf)
    logp_buf = torch.stack(logp_buf)
    rew_buf = torch.stack(rew_buf)
    val_buf = torch.stack(val_buf)
    done_buf = torch.stack(done_buf)
    trunc_buf = torch.stack(trunc_buf)

    # Compute advantages with GAE(λ)
    advantages = torch.zeros_like(rew_buf, device=device)
    returns = torch.zeros_like(rew_buf, device=device)
    gae = 0.0
    for t in reversed(range(horizon)):
        mask = 1.0 - torch.max(done_buf[t], trunc_buf[t])
        delta = rew_buf[t] + gamma * next_value * mask - val_buf[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = val_buf[t]
    returns = advantages + val_buf

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = {
        "obs": obs_buf,
        "actions": act_buf,
        "logp": logp_buf,
        "advantages": advantages,
        "returns": returns,
        "done": done_buf,
        "trunc": trunc_buf,
        "rewards": rew_buf,
    }
    return batch


def _make_env(env_type: str, debug_env: bool = False):
    if env_type == "goal_walk":
        from env_goal_walk import ThreeLPGoalWalkEnv  # lazy import to avoid pybind conflicts

        return ThreeLPGoalWalkEnv(debug_log=debug_env)
    elif env_type == "vel_walk":
        from env_vel_walk import ThreeLPVelWalkEnv

        return ThreeLPVelWalkEnv()
    else:
        from env_3lp import ThreeLPGotoGoalEnv  # lazy import

        return ThreeLPGotoGoalEnv(use_python_sim=False, debug_log=debug_env)


def _rollout_states(
    env,
    actor,
    device="cpu",
    max_steps=200,
    dense_stride=False,
    n_substeps=120,
    log_prefix="viz",
    override_params=None,
    override_phase_times=None,
):
    try:
        import threelp
    except Exception:
        threelp = None
    sim = env.sim
    states = []
    obs, _ = env.reset()
    sim = env.sim  # refresh in case reset recreated the sim
    states.append(sim.get_state() if not dense_stride else [float(v) for v in sim.get_state().q])
    if override_params is not None and hasattr(env, "sim") and env.sim is not None:
        try:
            env.sim.set_params(override_params)
        except Exception:
            pass
    if override_phase_times is not None and hasattr(env, "sim") and env.sim is not None:
        try:
            env.sim.set_phase_times(override_phase_times[0], override_phase_times[1])
        except Exception:
            pass
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    leg = sim.support_sign if hasattr(sim, "support_sign") else 1
    have_dense_foot = dense_stride and threelp is not None and hasattr(threelp, "simulate_stride_with_foot_offset")
    have_dense_inputs = dense_stride and threelp is not None and hasattr(threelp, "simulate_stride_with_inputs")
    if dense_stride and not (have_dense_foot or have_dense_inputs):
        print(f"[{log_prefix}] dense_stride requested but no dense samplers available; falling back to step states.")
        have_dense_foot = have_dense_inputs = False
    for _ in range(max_steps):
        with torch.no_grad():
            mean, _ = actor(obs_t)
            action = mean
            if env.action_space is not None:
                low = torch.as_tensor(env.action_space.low, device=action.device)
                high = torch.as_tensor(env.action_space.high, device=action.device)
                action = torch.max(torch.min(action, high), low)
        action_np = action.squeeze(0).cpu().numpy()
        use_dense_foot = have_dense_foot and action_np.shape[0] == 2
        use_dense_inputs = have_dense_inputs and action_np.shape[0] == 8
        if use_dense_foot:
            seg = threelp.simulate_stride_with_foot_offset(sim.get_state(), leg, action_np.tolist(), sim.t_ds, sim.t_ss, sim.get_params(), n_substeps)
            for s in seg:
                states.append(s if not dense_stride else [float(v) for v in s.q])
        elif use_dense_inputs:
            seg = threelp.simulate_stride_with_inputs(sim.get_state(), leg, action_np.tolist(), sim.t_ds, sim.t_ss, sim.get_params(), n_substeps)
            for s in seg:
                states.append(s if not dense_stride else [float(v) for v in s.q])
        obs, reward, done, trunc, info = env.step(action_np)
        leg = sim.support_sign if hasattr(sim, "support_sign") else leg
        if not (use_dense_foot or use_dense_inputs):
            states.append(sim.get_state() if not dense_stride else [float(v) for v in sim.get_state().q])
        if done or trunc:
            break
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return states


def maybe_visualize(env, actor, device="cpu", max_steps=200, loop=False, dense_stride=False, n_substeps=120, log_prefix="viz"):
    """
    Synchronous visualization: blocks until window is closed.
    """
    threelp_mod = None
    try:
        import threelp as tlp

        threelp_mod = tlp
    except Exception as e:
        print(f"[{log_prefix}] skip: threelp import failed ({e})")
    if threelp_mod is None or not hasattr(threelp_mod, "visualize_trajectory"):
        print(f"[{log_prefix}] skip: no visualize_trajectory available")
        return
    if not hasattr(env, "sim"):
        print(f"[{log_prefix}] skip: env has no sim")
        return
    sim = env.sim
    if sim is None:
        print(f"[{log_prefix}] skip: sim is None")
        return

    states = _rollout_states(env, actor, device=device, max_steps=max_steps, dense_stride=dense_stride, n_substeps=n_substeps, log_prefix=log_prefix)
    try:
        if dense_stride:
            # reconstruct state objs
            state_objs = []
            for q in states:
                s = threelp_mod.ThreeLPState()
                s.q = q
                state_objs.append(s)
            states_use = state_objs
        else:
            states_use = states
        print(f"[{log_prefix}] visualize with {len(states_use)} states, dense={dense_stride}")
        threelp_mod.visualize_trajectory(states_use, sim.t_ds, sim.t_ss, sim.get_params(), fps=60.0, loop=loop)
    except Exception as e:
        print(f"[{log_prefix}] visualize_trajectory error: {e}")


def _params_to_tuple(params):
    return (
        float(params.h1),
        float(params.h2),
        float(params.h3),
        float(params.wP),
        float(params.m1),
        float(params.m2),
        float(params.g),
    )


def _save_checkpoint(save_dir: Path, actor, critic, cfg, env_type: str, params_tuple, t_ds, t_ss, iteration: int, extra: dict | None = None):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "config": {
            "obs_dim": cfg.obs_dim,
            "action_dim": cfg.action_dim,
            "basis_dim": cfg.basis_dim,
            "encoder_type": cfg.encoder_type,
            "actor_basis_dim": cfg.actor_basis_dim,
            "critic_basis_dim": cfg.critic_basis_dim,
            "log_std_init": cfg.log_std_init,
            "hidden_sizes": list(cfg.hidden_sizes),
        },
        "env_type": env_type,
        "params": params_tuple,
        "t_ds": t_ds,
        "t_ss": t_ss,
        "iteration": iteration,
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, save_dir / "checkpoint.pt")
    meta = {
        "iteration": iteration,
        "env_type": env_type,
        "t_ds": t_ds,
        "t_ss": t_ss,
        "params": params_tuple,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _render_job(job):
    try:
        import threelp as tlp
        from policy_3lp import PolicyConfig, LinearBasisActor, MLPActor
    except Exception as e:
        print(f"[viz] render import error: {e}", flush=True)
        return

    try:
        env_type = job["env_type"]
        policy_type = job["policy_type"]
        cfg_dict = job["cfg"]
        state_dict = job["state_dict"]
        loop_flag = job["loop"]
        max_steps = job["max_steps"]
        dense_stride = job["dense_stride"]
        n_substeps = job["n_substeps"]
        t_ds_val = job["t_ds"]
        t_ss_val = job["t_ss"]
        params_tuple = job["params"]
        debug_env = job.get("debug_env", False)
    except Exception as e:
        print(f"[viz] render invalid job: {e}", flush=True)
        return

    def _tuple_to_params(p_tuple):
        p = tlp.ThreeLPParams()
        p.h1, p.h2, p.h3, p.wP, p.m1, p.m2, p.g = p_tuple
        return p

    try:
        env = _make_env(env_type, debug_env=debug_env)
    except Exception as e:
        print(f"[viz] render failed to build env ({env_type}): {e}", flush=True)
        return

    cfg_dict = dict(cfg_dict)
    if "hidden_sizes" in cfg_dict:
        cfg_dict["hidden_sizes"] = tuple(cfg_dict["hidden_sizes"])
    try:
        cfg = PolicyConfig(**cfg_dict)
    except Exception as e:
        print(f"[viz] render failed to rebuild config: {e}", flush=True)
        return

    try:
        if policy_type == "linear":
            actor = LinearBasisActor(cfg).to("cpu")
        elif policy_type == "mlp":
            actor = MLPActor(cfg).to("cpu")
        else:
            print(f"[viz] render: unknown policy_type {policy_type}", flush=True)
            return
        actor.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[viz] render failed to rebuild actor: {e}", flush=True)
        return

    try:
        state_lists = _rollout_states(
            env,
            actor,
            device="cpu",
            max_steps=max_steps,
            dense_stride=dense_stride,
            n_substeps=n_substeps,
            log_prefix="viz",
            override_params=_tuple_to_params(params_tuple),
            override_phase_times=(t_ds_val, t_ss_val),
        )
    except Exception as e:
        print(f"[viz] render rollout error: {e}", flush=True)
        return

    try:
        state_objs = []
        for q in state_lists:
            s = tlp.ThreeLPState()
            s.q = q if not hasattr(q, "q") else q.q
            state_objs.append(s)
        params_obj = _tuple_to_params(params_tuple)
        tlp.visualize_trajectory(
            state_objs,
            t_ds_val,
            t_ss_val,
            params_obj,
            fps=60.0,
            loop=loop_flag,
            wait_for_close=loop_flag,
        )
    except Exception as e:
        print(f"[viz] render visualize error: {e}", flush=True)


def _viz_worker(job_queue):
    try:
        import multiprocessing as mp_local
    except Exception as e:
        print(f"[viz] worker import error: {e}", flush=True)
        return

    render_proc = None
    ctx = mp_local.get_context("spawn")
    while True:
        try:
            job = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        except (EOFError, OSError, FileNotFoundError):
            break
        if job is None:
            if render_proc is not None and render_proc.is_alive():
                render_proc.terminate()
                render_proc.join(timeout=2.0)
            break
        if render_proc is not None and render_proc.is_alive():
            render_proc.terminate()
            render_proc.join(timeout=2.0)
        render_proc = ctx.Process(target=_render_job, args=(job,))
        render_proc.daemon = True
        render_proc.start()


class _VizProcessManager:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._queue = None

    def _start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._queue = self._ctx.Queue(maxsize=1)
        self._proc = self._ctx.Process(target=_viz_worker, args=(self._queue,))
        # Keep this non-daemonic so it can spawn a render subprocess.
        self._proc.daemon = False
        self._proc.start()

    def submit(self, job):
        self._start()
        if self._queue is None:
            return
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            pass

    def stop(self):
        if self._proc is None:
            return
        if self._proc.is_alive():
            if self._queue is not None:
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
                except Exception:
                    pass
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        self._proc = None
        self._queue = None


_viz_manager = None


def _get_viz_manager():
    global _viz_manager
    if _viz_manager is None:
        _viz_manager = _VizProcessManager()
        atexit.register(_viz_manager.stop)
    return _viz_manager


def maybe_visualize_async(
    env,
    actor,
    cfg,
    policy_type,
    env_type,
    device="cpu",
    max_steps=200,
    loop=False,
    dense_stride=False,
    n_substeps=120,
    log_prefix="viz",
    debug_env=False,
):
    """
    Fire-and-forget visualization in a child process to avoid blocking training.
    The worker reconstructs the env+actor and runs the rollout there, so the
    trainer never pauses. Uses a singleton worker process + queue so only one
    window is active; drops stale requests if training triggers a new
    visualization before the last finishes.
    """
    try:
        import threelp
    except Exception as e:
        print(f"[{log_prefix}] skip async: import error {e}")
        return
    if not hasattr(threelp, "visualize_trajectory"):
        print(f"[{log_prefix}] skip async: no visualize_trajectory")
        return
    if not hasattr(env, "sim"):
        print(f"[{log_prefix}] skip async: env has no sim")
        return
    sim = env.sim
    if not isinstance(sim, threelp.ThreeLPSim):
        print(f"[{log_prefix}] skip async: sim not threelp.ThreeLPSim")
        return

    # Snapshot actor weights to CPU to keep trainer device untouched.
    state_dict_cpu = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
    cfg_payload = {
        "obs_dim": cfg.obs_dim,
        "action_dim": cfg.action_dim,
        "basis_dim": cfg.basis_dim,
        "encoder_type": cfg.encoder_type,
        "actor_basis_dim": cfg.actor_basis_dim,
        "critic_basis_dim": cfg.critic_basis_dim,
        "log_std_init": cfg.log_std_init,
        "hidden_sizes": tuple(cfg.hidden_sizes),
    }

    job = {
        "env_type": env_type,
        "policy_type": policy_type,
        "cfg": cfg_payload,
        "state_dict": state_dict_cpu,
        "t_ds": sim.t_ds,
        "t_ss": sim.t_ss,
        "params": _params_to_tuple(sim.get_params()),
        "loop": loop,
        "max_steps": max_steps,
        "dense_stride": dense_stride,
        "n_substeps": n_substeps,
        "debug_env": debug_env,
    }

    mgr = _get_viz_manager()
    mgr.submit(job)


def train(
    policy_type: str = "linear",
    env_type: str = "goto_goal",
    ppo_epochs: int = 10,
    minibatch_size: int = 256,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    total_iterations: int = 1000,
    debug_env: bool = False,
    viz_every: int = 0,
    viz_loop: bool = False,
    viz_async: bool = False,
    save_dir: str | None = None,
    save_interval: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Make environment ---
    env = _make_env(env_type, debug_env=debug_env)

    # --- Build policy & critic ---
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    basis_dim = 64  # legacy MLP basis placeholder
    encoder_type = "goal_walk" if env_type == "goal_walk" else ("vel_walk" if env_type == "vel_walk" else "raw")

    actor_basis_dim = 15
    critic_basis_dim = 22
    if env_type == "goal_walk":
        actor_basis_dim = 15
        critic_basis_dim = 22
    elif env_type == "vel_walk":
        actor_basis_dim = 16
        critic_basis_dim = 45

    cfg = PolicyConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        basis_dim=basis_dim,
        encoder_type=encoder_type,
        actor_basis_dim=actor_basis_dim,
        critic_basis_dim=critic_basis_dim,
    )

    if policy_type == "linear":
        actor = LinearBasisActor(cfg).to(device)
        critic = QuadraticCritic(basis_dim=cfg.critic_basis_dim).to(device)
    elif policy_type == "mlp":
        actor = MLPActor(cfg).to(device)
        critic = MLPCritic(cfg).to(device)
    else:
        raise ValueError(f"Unknown policy_type {policy_type}")

    pi_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    v_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    obs_norm = RunningNorm(shape=(obs_dim,), device=device)

    for iteration in range(total_iterations):
        if iteration == 0 and viz_every > 0:
            print(f"[viz] enabled: viz_every={viz_every} loop={viz_loop} async={viz_async}")
        batch = collect_rollout(env, actor, critic, horizon=2048, device=device, obs_norm=obs_norm)

        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        old_logp = batch["logp"].to(device)
        advantages = batch["advantages"].to(device)
        returns = batch["returns"].to(device)
        rewards_np = batch["rewards"].cpu().numpy()
        done_np = batch["done"].cpu().numpy()
        trunc_np = batch["trunc"].cpu().numpy()

        # PPO mini-batch updates
        N = obs.shape[0]
        entropy_meter = 0.0
        clip_frac_meter = 0.0
        pi_loss_meter = 0.0
        v_loss_meter = 0.0
        total_batches = 0
        # Track simple rollout stats
        episode_rewards = []
        episode_lengths = []
        ep_reward = 0.0
        ep_len = 0
        for r, d, tr in zip(rewards_np, done_np, trunc_np):
            ep_reward += r
            ep_len += 1
            if d or tr:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_len)
                ep_reward = 0.0
                ep_len = 0

        for _ in range(ppo_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, minibatch_size):
                end = start + minibatch_size
                idx = perm[start:end]

                obs_mb = obs[idx]
                act_mb = actions[idx]
                old_logp_mb = old_logp[idx]
                adv_mb = advantages[idx]
                ret_mb = returns[idx]

                mean, log_std = actor(obs_mb)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act_mb).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (logp - old_logp_mb).exp()
                unclipped = ratio * adv_mb
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                pi_loss = -torch.min(unclipped, clipped).mean() - entropy_coef * entropy

                if hasattr(actor, "encoder") and hasattr(actor.encoder, "critic_features"):
                    critic_phi = actor.encoder.critic_features(obs_mb)
                    values = critic(critic_phi)
                else:
                    values = critic(obs_mb)
                v_loss = nn.functional.mse_loss(values, ret_mb)

                pi_optimizer.zero_grad()
                pi_loss.backward()
                pi_optimizer.step()

                v_optimizer.zero_grad()
                v_loss.backward()
                v_optimizer.step()

                clip_frac = (torch.gt(ratio, 1.0 + clip_eps) | torch.lt(ratio, 1.0 - clip_eps)).float().mean()

                entropy_meter += entropy.item()
                clip_frac_meter += clip_frac.item()
                pi_loss_meter += pi_loss.item()
                v_loss_meter += v_loss.item()
                total_batches += 1

        if iteration % 10 == 0:
            avg_entropy = entropy_meter / max(1, total_batches)
            avg_clip = clip_frac_meter / max(1, total_batches)
            avg_pi = pi_loss_meter / max(1, total_batches)
            avg_v = v_loss_meter / max(1, total_batches)
            avg_ep_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            avg_ep_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            print(
                f"Iter {iteration:04d} | "
                f"pi {avg_pi:.3f} | "
                f"v {avg_v:.3f} | "
                f"ent {avg_entropy:.3f} | "
                f"clip {avg_clip:.3f} | "
                f"adv {advantages.mean().item():.3f} | "
                f"R_ep {avg_ep_rew:.3f} | "
                f"len {avg_ep_len:.1f}"
            )
        if viz_every > 0 and iteration > 0 and iteration % viz_every == 0:
            # Use dense stride viz when samplers are available (2D or 8D actions).
            dense_ok = hasattr(env, "action_space") and env.action_space.shape[0] in (2, 8)
            print(f"[viz] trigger at iter {iteration} | dense={dense_ok} | async={viz_async}")
            print(f"[viz] sim type: {type(env.sim)}")
            try:
                if viz_async:
                    maybe_visualize_async(
                        env,
                        actor,
                        cfg,
                        policy_type,
                        env_type,
                        device=device,
                        max_steps=200,
                        loop=viz_loop,
                        dense_stride=dense_ok,
                        debug_env=debug_env,
                    )
                else:
                    maybe_visualize(env, actor, device=device, max_steps=200, loop=viz_loop, dense_stride=dense_ok)
            except Exception as e:
                print(f"[viz] error dispatching visualize: {e}")

        if save_dir and ((save_interval > 0 and iteration % save_interval == 0) or iteration == total_iterations - 1):
            ckpt_dir = Path(save_dir)
            params_tuple = _params_to_tuple(env.sim.get_params()) if hasattr(env, "sim") and env.sim is not None else None
            _save_checkpoint(ckpt_dir, actor, critic, cfg, env_type, params_tuple, getattr(env.sim, "t_ds", None), getattr(env.sim, "t_ss", None), iteration)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train 3LP policy with PPO-style updates.")
    parser.add_argument("--policy-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--env-type", choices=["goto_goal", "goal_walk", "vel_walk"], default="goto_goal")
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--debug-env", action="store_true")
    parser.add_argument("--viz-every", type=int, default=0, help="If >0 and visualize_trajectory is available, render a rollout every N iterations.")
    parser.add_argument("--viz-loop", action="store_true", help="Keep the visualizer window open and replay when visualizing.")
    parser.add_argument("--viz-async", type=int, choices=[0, 1], default=0, help="Run visualization in a background process (1) or synchronously (0).")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to write checkpoints.")
    parser.add_argument("--save-interval", type=int, default=0, help="Save a checkpoint every N iterations (and also at the final iteration).")
    args = parser.parse_args()

    train(
        policy_type=args.policy_type,
        env_type=args.env_type,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        total_iterations=args.iterations,
        debug_env=args.debug_env,
        viz_every=args.viz_every,
        viz_loop=args.viz_loop,
        viz_async=bool(args.viz_async),
        save_dir=args.save_dir,
        save_interval=args.save_interval,
    )
