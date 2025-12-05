import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from env_threelp_stride_sac import ThreeLPStrideEnv


# --------------------------------------------------------------------------- #
#  Models & Buffer                                                            #
# --------------------------------------------------------------------------- #

class ReplayBuffer:
    """Simple FIFO replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.constant_(m.bias, 0.0)


class SoftQNetwork(nn.Module):
    """Twin Q-network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(_weights_init)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class Actor(nn.Module):
    """Gaussian policy with Tanh squashing."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, action_space=None):
        super().__init__()
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, act_dim)
        self.log_std_linear = nn.Linear(hidden_dim, act_dim)
        self.apply(_weights_init)

        if action_space is None:
            action_scale = torch.tensor(1.0)
            action_bias = torch.tensor(0.0)
        else:
            high = torch.as_tensor(action_space.high, dtype=torch.float32)
            low = torch.as_tensor(action_space.low, dtype=torch.float32)
            action_scale = (high - low) / 2.0
            action_bias = (high + low) / 2.0

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def make_env(args, reference_cache=None) -> ThreeLPStrideEnv:
    """Build a stride-level SAC env with the current CLI configuration."""
    v_range = (
        (args.v_cmd, args.v_cmd) if args.v_cmd is not None else (args.v_cmd_min, args.v_cmd_max)
    )
    env = ThreeLPStrideEnv(
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        inner_dt=args.inner_dt,
        max_strides=args.max_strides,
        action_scale=args.action_scale,
        alive_bonus=args.alive_bonus,
        q_e_diag=tuple(args.q_e_diag),
        q_v=args.q_v,
        r_action=args.r_action,
        terminal_penalty=args.terminal_penalty,
        fall_bounds=tuple(args.fall_bounds),
        v_cmd_range=v_range,
        reset_noise_std=args.reset_noise_std,
        ref_substeps=args.ref_substeps,
        reference_cache=reference_cache,
        obs_clip=args.obs_clip,
    )
    return env


def evaluate_policy(
    actor: Actor,
    env: ThreeLPStrideEnv,
    device: torch.device,
    episodes: int = 3,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Average return and length over a few evaluation episodes."""
    returns = []
    lengths = []

    for ep in range(episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            with torch.no_grad():
                s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                action, _, mean = actor.sample(s_t)
                act = mean if deterministic else action
                act_np = act.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated
            ep_ret += reward
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

    return float(np.mean(returns)), float(np.mean(lengths))


def _maybe_precache(env: ThreeLPStrideEnv, commands: Iterable[float]) -> None:
    """Warm up reference cache so the first episodes don't pay the cost."""
    for v in commands:
        try:
            env._get_reference(float(v))
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"[precache] failed to build reference for v_cmd={v}: {exc}")


# --------------------------------------------------------------------------- #
#  Training                                                                   #
# --------------------------------------------------------------------------- #

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"stride_sac_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_csv = run_dir / "metrics.csv"
    model_path = Path(args.save_path) if args.save_path else run_dir / "actor.pt"

    shared_cache = {}
    env = make_env(args, reference_cache=shared_cache)
    eval_env = make_env(args, reference_cache=shared_cache)

    if args.precache_cmds:
        _maybe_precache(env, args.precache_cmds)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim, hidden_dim=args.hidden_dim, action_space=env.action_space).to(device)
    q_net = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    q_target = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q_optim = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(act_dim)

    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device=device)

    with log_csv.open("w", encoding="utf-8") as f:
        f.write("step,episode_return,episode_len,q_loss,actor_loss,alpha,eval_return,eval_len,v_cmd\n")

    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"[init] device={device}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"[init] logging to {log_csv}")

    obs, info = env.reset(seed=args.seed)
    obs = obs.astype(np.float32)
    ep_return = 0.0
    ep_len = 0
    ep_idx = 0
    last_q_loss = 0.0
    last_actor_loss = 0.0
    last_eval_ret = np.nan
    last_eval_len = np.nan

    for global_step in range(1, args.total_timesteps + 1):
        if global_step <= args.start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                a_t, _, _ = actor.sample(s_t)
                action = a_t.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        real_done = float(terminated)

        buffer.add(obs, action, reward, next_obs, real_done)
        obs = next_obs.astype(np.float32)
        ep_return += reward
        ep_len += 1

        if buffer.size >= args.batch_size and global_step > args.start_steps:
            for _ in range(args.updates_per_step):
                b_s, b_a, b_r, b_ns, b_d = buffer.sample(args.batch_size)

                with torch.no_grad():
                    next_action, next_log_pi, _ = actor.sample(b_ns)
                    q1_next, q2_next = q_target(b_ns, next_action)
                    min_q_next = torch.min(q1_next, q2_next) - log_alpha.exp() * next_log_pi
                    target_q = b_r + (1 - b_d) * args.gamma * min_q_next

                q1, q2 = q_net(b_s, b_a)
                q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

                pi, log_pi, _ = actor.sample(b_s)
                q1_pi, q2_pi = q_net(b_s, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (log_alpha.exp() * log_pi - min_q_pi).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                alpha_loss = -(log_alpha.exp() * (log_pi + target_entropy).detach()).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                for param, target_param in zip(q_net.parameters(), q_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

                last_q_loss = q_loss.item()
                last_actor_loss = actor_loss.item()

        if done:
            if (ep_idx + 1) % args.log_interval == 0:
                alpha_val = float(log_alpha.exp().item())
                cmd = float(info.get("v_cmd", np.nan))
                print(
                    f"[step {global_step}] ep={ep_idx} ret={ep_return:.2f} len={ep_len} "
                    f"alpha={alpha_val:.3f} v_cmd={cmd:.2f}"
                )

            with log_csv.open("a", encoding="utf-8") as f:
                alpha_val = float(log_alpha.exp().item())
                cmd = float(info.get("v_cmd", np.nan))
                f.write(
                    f"{global_step},{ep_return},{ep_len},{last_q_loss},{last_actor_loss},"
                    f"{alpha_val},{last_eval_ret},{last_eval_len},{cmd}\n"
                )

            obs, info = env.reset()
            obs = obs.astype(np.float32)
            ep_return = 0.0
            ep_len = 0
            ep_idx += 1

        if args.eval_every > 0 and global_step % args.eval_every == 0:
            last_eval_ret, last_eval_len = evaluate_policy(
                actor, eval_env, device, episodes=args.eval_episodes, deterministic=True, seed=args.seed
            )
            print(
                f"[eval @ {global_step}] avg_return={last_eval_ret:.2f} avg_len={last_eval_len:.2f}"
            )

        if args.save_every > 0 and global_step % args.save_every == 0:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"actor_state_dict": actor.state_dict()}, model_path)
            print(f"[save] checkpointed actor to {model_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor.state_dict()}, model_path)
    print(f"[done] saved actor to {model_path}")


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="SAC trainer for ThreeLP stride-level env.")

    # Env configuration
    p.add_argument("--t-ds", type=float, default=0.1, help="Double-support duration.")
    p.add_argument("--t-ss", type=float, default=0.6, help="Single-support duration.")
    p.add_argument("--inner-dt", type=float, default=0.005, help="Simulator integration dt inside each stride.")
    p.add_argument("--max-strides", type=int, default=200, help="Episode length in strides.")
    p.add_argument("--action-scale", type=float, default=40.0, help="Scale on tanh(action) residuals.")
    p.add_argument("--alive-bonus", type=float, default=5.0, help="Reward bonus per simulated second.")
    p.add_argument("--q-e-diag", type=float, nargs=8, default=(20.0, 20.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0))
    p.add_argument("--q-v", type=float, default=2.0)
    p.add_argument("--r-action", type=float, default=1e-3)
    p.add_argument("--terminal-penalty", type=float, default=50.0)
    p.add_argument("--fall-bounds", type=float, nargs=4, default=(1.0, 0.5, 10.0, 10.0))
    p.add_argument("--reset-noise-std", type=float, default=0.01)
    p.add_argument("--obs-clip", type=float, default=1e3)
    p.add_argument("--v-cmd", type=float, default=None, help="Fix command speed; overrides range if set.")
    p.add_argument("--v-cmd-min", type=float, default=0.6)
    p.add_argument("--v-cmd-max", type=float, default=1.4)
    p.add_argument("--ref-substeps", type=int, default=None, help="Override reference sampling resolution.")
    p.add_argument(
        "--precache-cmds",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of command speeds to pre-build references for.",
    )

    # SAC configuration
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--start-steps", type=int, default=10_000, help="Steps of random policy before SAC updates.")
    p.add_argument("--updates-per-step", type=int, default=1)

    # Logging / eval / misc
    p.add_argument("--log-interval", type=int, default=10, help="Episodes between stdout logs.")
    p.add_argument("--eval-every", type=int, default=25_000, help="Steps between eval runs (0 to disable).")
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--save-every", type=int, default=0, help="Checkpoint every N steps (0 to disable).")
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None, help="Root folder for logs/checkpoints.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action="store_true", default=False)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
