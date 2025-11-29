# train_3lp_ac.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_3lp import ThreeLPGotoGoalEnv
from policy_3lp import PolicyConfig, LinearBasisActor, QuadraticCritic


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
            phi = actor.encoder(obs_n.unsqueeze(0))
            value = critic(phi).squeeze(0)
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
        phi_last = actor.encoder(obs_n.unsqueeze(0))
        next_value = critic(phi_last).squeeze(0)

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
    }
    return batch


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Make environment ---
    env = ThreeLPGotoGoalEnv(use_python_sim=False)

    # --- Build policy & critic ---
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    basis_dim = 64  # choose something reasonable

    cfg = PolicyConfig(obs_dim=obs_dim, action_dim=action_dim, basis_dim=basis_dim)

    actor = LinearBasisActor(cfg).to(device)
    critic = QuadraticCritic(basis_dim=basis_dim).to(device)

    pi_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    v_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    obs_norm = RunningNorm(shape=(obs_dim,), device=device)

    for iteration in range(1000):
        batch = collect_rollout(env, actor, critic, horizon=2048, device=device, obs_norm=obs_norm)

        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        old_logp = batch["logp"].to(device)
        advantages = batch["advantages"].to(device)
        returns = batch["returns"].to(device)

        # Recompute dist and log probs under current policy
        mean, log_std = actor(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()

        # PPO-style clipped policy loss
        ratio = (logp - old_logp).exp()
        clip_eps = 0.2
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        pi_loss = -torch.min(unclipped, clipped).mean() - 0.01 * entropy

        # Critic loss
        phi = actor.encoder(obs)
        values = critic(phi)
        v_loss = nn.functional.mse_loss(values, returns)

        # Optimize actor
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        # Optimize critic
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

        if iteration % 10 == 0:
            print(
                f"Iter {iteration:04d} | "
                f"pi_loss {pi_loss.item():.3f} | "
                f"v_loss {v_loss.item():.3f} | "
                f"adv {advantages.mean().item():.3f}"
            )


if __name__ == "__main__":
    train()
