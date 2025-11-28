# train_3lp_ac.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_3lp import ThreeLPGotoGoalEnv
from policy_3lp import PolicyConfig, LinearBasisActor, QuadraticCritic


def collect_rollout(env, actor, critic, horizon=2048, gamma=0.99, lam=0.95, device="cpu"):
    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    val_buf = []
    done_buf = []

    obs, info = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    for t in range(horizon):
        with torch.no_grad():
            # Build basis and value
            phi = actor.encoder(obs.unsqueeze(0))
            value = critic(phi).squeeze(0)
            action, logp, _ = actor.act(obs.unsqueeze(0))
        action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, done, truncated, info = env.step(action_np)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        obs_buf.append(obs)
        act_buf.append(action.squeeze(0))
        logp_buf.append(logp.squeeze(0))
        rew_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
        val_buf.append(value)
        done_buf.append(torch.tensor(done, dtype=torch.float32, device=device))

        obs = next_obs_t
        if done:
            obs, info = env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    # Convert to tensors
    obs_buf = torch.stack(obs_buf)
    act_buf = torch.stack(act_buf)
    logp_buf = torch.stack(logp_buf)
    rew_buf = torch.stack(rew_buf)
    val_buf = torch.stack(val_buf)
    done_buf = torch.stack(done_buf)

    # Compute advantages with GAE(λ)
    advantages = torch.zeros_like(rew_buf, device=device)
    returns = torch.zeros_like(rew_buf, device=device)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(horizon)):
        mask = 1.0 - done_buf[t]
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
    env = ThreeLPGotoGoalEnv()

    # --- Build policy & critic ---
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    basis_dim = 64  # choose something reasonable

    cfg = PolicyConfig(obs_dim=obs_dim, action_dim=action_dim, basis_dim=basis_dim)

    actor = LinearBasisActor(cfg).to(device)
    critic = QuadraticCritic(basis_dim=basis_dim).to(device)

    pi_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    v_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    for iteration in range(1000):
        batch = collect_rollout(env, actor, critic, horizon=2048, device=device)

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

        # Policy loss (simple A2C-style, no clipping)
        pi_loss = -(logp * advantages).mean()

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
