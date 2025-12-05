import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from env_high_rate_3lp import ThreeLPHighRateEnv


class ReplayBuffer:
    """Simple replay buffer for off-policy learning."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.device = device

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
            torch.as_tensor(self.obs[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.actions[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.next_obs[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.dones[idx], device=self.device, dtype=torch.float32),
        )


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.constant_(m.bias, 0.0)


class SoftQNetwork(nn.Module):
    """Twin Q-network for SAC."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.linear1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.linear4 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

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

    def __init__(self, obs_dim, act_dim, hidden_dim=256, action_space=None):
        super().__init__()
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, act_dim)
        self.log_std_linear = nn.Linear(hidden_dim, act_dim)
        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            self.action_bias = torch.as_tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def preprocess_obs(env, obs, v_nom):
    """
    Map raw env observation -> shaped features:
    [scaled error(8), sinφ, cosφ, v_cmd - v_nom, torque error(8)] -> R^19
    """
    x = obs[:8]
    phi = obs[8]
    v_cmd = obs[9]
    p_err = obs[10:]

    x_ref = env.reference_state(phi) if hasattr(env, "reference_state") else np.zeros_like(x)
    e = x - x_ref
    sin_phi = np.sin(2 * np.pi * phi)
    cos_phi = np.cos(2 * np.pi * phi)

    z = np.concatenate([5.0 * e, [sin_phi, cos_phi, v_cmd - v_nom], p_err])
    if not np.all(np.isfinite(z)):
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z


def train_sac(args):
    fixed_speed = 1.1
    env = ThreeLPHighRateEnv(
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        dt=args.dt,
        max_steps=args.max_env_steps,
        action_clip=100.0,
        alpha_p= 1.0,#0.2,
        p_decay= 1.0,#0.995,
        alive_bonus=10.0,
        v_cmd_range=(fixed_speed, fixed_speed),
        random_phase=True,
        fall_bounds=(1.0, 0.5, 10.0, 10.0),
        q_e_diag=(2.0, 2.0, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1),
        q_v=0.5,
        r_u=0.001,
        reset_noise_std=0.0 #0.1, # set to 0 to start to see if we get anything
    )
    v_nom = fixed_speed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    obs_dim = 19
    act_dim = env.action_dim

    actor = Actor(obs_dim, act_dim, hidden_dim=args.hidden_dim, action_space=env.action_space).to(device)
    qf = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    qf_target = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    qf_target.load_state_dict(qf.state_dict())

    q_optimizer = optim.Adam(qf.parameters(), lr=args.lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)

    target_entropy = -torch.prod(torch.tensor(env.action_space.shape, device=device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.lr)

    replay_buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("step,ep_ret,ep_len,q_loss,actor_loss,alpha\n")

    print(f"Starting SAC on {device}. Target entropy: {target_entropy:.2f}")

    state_np, _ = env.reset(seed=args.seed)
    state = preprocess_obs(env, state_np, v_nom)

    ep_ret, ep_len, ep_count = 0.0, 0, 0
    last_q_loss, last_actor_loss = 0.0, 0.0

    for global_step in range(1, args.total_timesteps + 1):
        if global_step <= args.start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                a_t, _, _ = actor.sample(s_t)
                action = a_t.cpu().numpy()[0]

        next_state_np, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess_obs(env, next_state_np, v_nom)
        real_done = float(terminated)

        replay_buffer.add(state, action, reward, next_state, real_done)
        state = next_state
        ep_ret += reward
        ep_len += 1

        if (
            replay_buffer.size >= args.batch_size
            and global_step >= args.start_steps
        ):
            for _ in range(args.updates_per_step):
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(args.batch_size)

                with torch.no_grad():
                    next_action, next_log_pi, _ = actor.sample(b_ns)
                    q1_next, q2_next = qf_target(b_ns, next_action)
                    min_q_next = torch.min(q1_next, q2_next) - log_alpha.exp() * next_log_pi
                    target_q = b_r + (1 - b_d) * args.gamma * min_q_next

                q1, q2 = qf(b_s, b_a)
                qf_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                pi, log_pi, _ = actor.sample(b_s)
                q1_pi, q2_pi = qf(b_s, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (log_alpha.exp() * log_pi - min_q_pi).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                alpha_loss = -(log_alpha.exp() * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

                last_q_loss = qf_loss.item()
                last_actor_loss = actor_loss.item()

        if done:
            if (ep_count + 1) % 10 == 0:
                print(
                    f"Step {global_step} | Ep {ep_count} | Ret: {ep_ret:.2f} | "
                    f"Len: {ep_len} | Alpha: {log_alpha.exp().item():.3f}"
                )

            with open(log_path, "a") as f:
                f.write(
                    f"{global_step},{ep_ret},{ep_len},{last_q_loss},"
                    f"{last_actor_loss},{log_alpha.exp().item()}\n"
                )

            state_np, _ = env.reset()
            state = preprocess_obs(env, state_np, v_nom)
            ep_ret, ep_len = 0.0, 0
            ep_count += 1

    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(actor.state_dict(), args.save_path)
        print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--start-steps", type=int, default=1_000_000) #default=10_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    #parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=0.999) # temp -> increase future reward contribution

    #parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.01) # need to figure out what this is


    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--t-ds", type=float, default=0.1)
    parser.add_argument("--t-ss", type=float, default=0.6)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--max-env-steps", type=int, default=2000)

    parser.add_argument("--log-path", type=str, default="logs/sac_log.csv")
    parser.add_argument("--save-path", type=str, default="weights/sac_model.pt")

    args = parser.parse_args()
    train_sac(args)
