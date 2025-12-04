import argparse
import time
from pathlib import Path
from typing import Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from env_high_rate_3lp import ThreeLPHighRateEnv


# --- Neural Networks ---

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Critic: Estimates V(s)
        self.critic = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, 1), std=1.0)
        )

        # Actor: Estimates mean action mu(s)
        self.actor = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)
        )

        # Learnable log_std for exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu = self.actor(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(x)

        return action, log_prob, entropy, value


# --- Feature Engineering ---

def preprocess_obs(env, obs, v_nom):
    """
    Converts raw environment observation into the 'z' feature vector
    expected by the network.

    Raw Obs (10): [s1... (8), phi, v_cmd]
    Output z (11): [5.0 * error (8), sin(phi), cos(phi), v_err]
    """
    x = obs[:8]
    phi = obs[8]
    v_cmd = obs[9]

    if hasattr(env, "reference_state"):
        x_ref = env.reference_state(phi)
    else:
        x_ref = np.zeros_like(x)

    e = x - x_ref

    sin_phi = np.sin(2 * np.pi * phi)
    cos_phi = np.cos(2 * np.pi * phi)

    # Moderate scaling on error to help NN conditioning
    z = np.concatenate([5.0 * e, [sin_phi, cos_phi, v_cmd - v_nom]])
    return torch.FloatTensor(z)


# --- PPO Trainer ---

def train_ppo(args):
    # Curriculum: fix speed to a single value for initial learning.
    fixed_speed = 1.2 #0.75
    env = ThreeLPHighRateEnv(
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        dt=args.dt,
        max_steps=args.max_env_steps,
        action_clip=100.0,
        alpha_p=0.05,
        p_decay=0.98,
        alive_bonus=10.0,
        v_cmd_range=(fixed_speed, fixed_speed),
        fall_bounds=(5.0, 5.0, 100.0, 100.0),
        reset_noise_std=0.0,
    )
    v_nom = fixed_speed

    obs_dim = 11  # Dimension of 'z' vector
    act_dim = env.action_dim

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = ActorCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Logging
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("update,ep_return,ep_length,value_loss,policy_loss\n")

    # Buffers
    obs = torch.zeros((args.num_steps, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, act_dim)).to(device)
    logprobs = torch.zeros((args.num_steps)).to(device)
    rewards = torch.zeros((args.num_steps)).to(device)
    dones = torch.zeros((args.num_steps)).to(device)
    values = torch.zeros((args.num_steps)).to(device)

    # Tracking
    global_step = 0
    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs = preprocess_obs(env, next_obs_np, v_nom).to(device)
    next_done = torch.zeros(1).to(device)

    ep_ret = 0
    ep_len = 0
    ep_count = 0

    num_updates = args.total_timesteps // args.num_steps
    print(f"Starting PPO on {device}: {num_updates} updates, {args.num_steps} steps per update, v_cmd={fixed_speed}")

    for update in range(1, num_updates + 1):
        # --- 1. Data Collection (Rollout) ---
        agent.eval()
        for step in range(args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Step Env
            action_np = action.cpu().numpy()
            next_obs_np, reward, term, trunc, _ = env.step(action_np)

            # Use raw reward; env now provides positive alive bonus.
            rewards[step] = torch.tensor(reward).to(device)

            next_obs = preprocess_obs(env, next_obs_np, v_nom).to(device)
            done_bool = term or trunc
            next_done = torch.tensor(float(done_bool)).to(device)

            ep_ret += reward
            ep_len += 1

            if done_bool:
                print(f"[Ep {ep_count}] Return: {ep_ret:.2f} Length: {ep_len}")
                ep_ret = 0
                ep_len = 0
                ep_count += 1
                next_obs_np, _ = env.reset()
                next_obs = preprocess_obs(env, next_obs_np, v_nom).to(device)

        # --- 2. Advantage Estimation (GAE) ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        # --- 3. Policy Optimization ---
        agent.train()
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, act_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing epochs
        clip_fracs = []
        for epoch in range(args.update_epochs):
            # Generate random indices
            b_inds = np.arange(args.num_steps)
            np.random.shuffle(b_inds)

            for start in range(0, args.num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Normalize Advantage
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss (Clipped)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (Clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # --- 4. Logging ---
        with open(log_path, "a") as f:
            f.write(f"{update},{ep_ret},{ep_len},{v_loss.item():.4f},{pg_loss.item():.4f}\n")

        if update % 10 == 0:
            print(f"Update {update}/{num_updates} | Loss: {loss.item():.4f} | Val: {v_loss.item():.4f}")

    # Save
    if args.save_path:
        torch.save(agent.state_dict(), args.save_path)
        print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="total timesteps of the experiments")
    parser.add_argument("--lr", type=float, default=3e-4, help="the learning rate of the optimizer")

    # Env params
    parser.add_argument("--t-ds", type=float, default=0.1)
    parser.add_argument("--t-ss", type=float, default=0.6)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-env-steps", type=int, default=2000)

    # PPO params
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--minibatch-size", type=int, default=64, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

    # Network
    parser.add_argument("--hidden-dim", type=int, default=64)

    # Logging
    parser.add_argument("--log-path", type=str, default="logs/ppo_log.csv")
    parser.add_argument("--save-path", type=str, default="weights/ppo_model.pt")

    args = parser.parse_args()
    train_ppo(args)
