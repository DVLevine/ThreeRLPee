# train_3lp_ac.py
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


def maybe_visualize(env, actor, device="cpu", max_steps=200, loop=False):
    """
    Synchronous visualization: blocks until window is closed.
    """
    try:
        import threelp
    except Exception:
        return
    if not hasattr(threelp, "visualize_trajectory"):
        return
    if not hasattr(env, "sim"):
        return
    sim = env.sim
    if not isinstance(sim, threelp.ThreeLPSim):
        return

    states = []
    # Capture current sim state
    states.append(sim.get_state())
    obs, _ = env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(max_steps):
        with torch.no_grad():
            mean, log_std = actor(obs_t)
            action = mean  # deterministic mean
            # Clip to action space
            if env.action_space is not None:
                low = torch.as_tensor(env.action_space.low, device=action.device)
                high = torch.as_tensor(env.action_space.high, device=action.device)
                action = torch.max(torch.min(action, high), low)
        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, done, trunc, info = env.step(action_np)
        states.append(sim.get_state())
        if done or trunc:
            break
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    try:
        threelp.visualize_trajectory(states, sim.t_ds, sim.t_ss, sim.get_params(), fps=60.0, loop=loop)
    except Exception:
        pass


def maybe_visualize_async(env, actor, device="cpu", max_steps=200, loop=False):
    """
    Fire-and-forget visualization in a child process to avoid blocking training.
    Converts states to plain lists for pickling; reconstructs in the child.
    """
    try:
        import multiprocessing as mp
        import threelp
    except Exception:
        return
    if not hasattr(threelp, "visualize_trajectory"):
        return
    if not hasattr(env, "sim"):
        return
    sim = env.sim
    if not isinstance(sim, threelp.ThreeLPSim):
        return

    # Rollout policy deterministically
    states = []
    states.append([float(v) for v in sim.get_state().q])
    obs, _ = env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(max_steps):
        with torch.no_grad():
            mean, _ = actor(obs_t)
            action = mean
            if env.action_space is not None:
                low = torch.as_tensor(env.action_space.low, device=action.device)
                high = torch.as_tensor(env.action_space.high, device=action.device)
                action = torch.max(torch.min(action, high), low)
        obs, reward, done, trunc, info = env.step(action.squeeze(0).cpu().numpy())
        states.append([float(v) for v in env.sim.get_state().q])
        if done or trunc:
            break
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    t_ds = sim.t_ds
    t_ss = sim.t_ss
    params = sim.get_params()

    def _worker(state_lists, t_ds_val, t_ss_val, params_obj, loop_flag):
        try:
            import threelp as tlp
            state_objs = []
            for q in state_lists:
                s = tlp.ThreeLPState()
                s.q = q
                state_objs.append(s)
            tlp.visualize_trajectory(state_objs, t_ds_val, t_ss_val, params_obj, fps=60.0, loop=loop_flag)
        except Exception:
            pass

    p = mp.Process(target=_worker, args=(states, t_ds, t_ss, params, loop))
    p.daemon = True
    p.start()


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
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Make environment ---
    if env_type == "goal_walk":
        from env_goal_walk import ThreeLPGoalWalkEnv  # lazy import to avoid pybind conflicts

        env = ThreeLPGoalWalkEnv(debug_log=debug_env)
    elif env_type == "vel_walk":
        from env_vel_walk import ThreeLPVelWalkEnv

        env = ThreeLPVelWalkEnv()
    else:
        from env_3lp import ThreeLPGotoGoalEnv  # lazy import

        env = ThreeLPGotoGoalEnv(use_python_sim=False, debug_log=debug_env)

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
            if viz_async:
                maybe_visualize_async(env, actor, device=device, max_steps=200, loop=viz_loop)
            else:
                maybe_visualize(env, actor, device=device, max_steps=200, loop=viz_loop)


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
    parser.add_argument("--viz-async", action="store_true", help="Run visualization in a background process to avoid blocking training.")
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
        viz_async=args.viz_async,
    )
