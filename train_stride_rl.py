"""
Canonical stride-level actor–critic training (Phase A) using torch autograd.

Features:
- Linear Gaussian actor over φ_a = [δx (8), command (1), bias].
- Fixed diagonal exploration std.
- Quadratic-monomial critic updated with semi-gradient TD(0).
- No observation normalization (physics-scale features are required for linear policies).
"""
import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from env_canonical_stride import ThreeLPCanonicalStrideEnv


def phi_actor(obs: np.ndarray) -> np.ndarray:
    """Actor features φ_a = [δx, command, 1]."""
    return np.concatenate([obs.astype(np.float64), np.array([1.0], dtype=np.float64)], axis=0)


def phi_critic(z: np.ndarray) -> np.ndarray:
    """All unique quadratic monomials of z (upper triangle)."""
    feats = []
    d = z.shape[0]
    for i in range(d):
        for j in range(i, d):
            feats.append(z[i] * z[j])
    return np.asarray(feats, dtype=np.float64)


class LinearGaussianActor(nn.Module):
    """
    Linear Gaussian policy: a ~ N(theta * phi, diag(sigma^2)), fixed sigma.
    """

    def __init__(self, d_s: int, init_std: float = 0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros((8, d_s), dtype=torch.float32))
        log_std = math.log(init_std)
        self.register_buffer("log_std", torch.full((8,), float(log_std), dtype=torch.float32))

    def forward(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # phi: (..., d_s)
        mean = torch.matmul(self.theta, phi)
        return mean, self.log_std

    def sample(
        self, phi: torch.Tensor, rng: torch.Generator | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(phi)
        std = log_std.exp()
        noise = torch.randn(mean.shape, device=mean.device, generator=rng) * std
        action = mean + noise
        return action, mean, log_std


@dataclass
class QuadraticCritic:
    w: np.ndarray  # (d_c,)

    @classmethod
    def zeros(cls, d_c: int):
        return cls(np.zeros(d_c, dtype=np.float64))

    def value(self, z: np.ndarray) -> float:
        return float(self.w @ phi_critic(z))

    def update(self, delta: float, z: np.ndarray, alpha_c: float):
        self.w += alpha_c * delta * phi_critic(z)


def train(
    episodes: int,
    gamma: float,
    alpha_a: float,
    alpha_c: float,
    init_std: float,
    seed: int | None,
    log_interval: int,
    env_kwargs: dict,
    resample_command: bool = False,
):
    rng_np = np.random.default_rng(seed)
    torch_rng = torch.Generator().manual_seed(seed if seed is not None else 0)

    env = ThreeLPCanonicalStrideEnv(**env_kwargs, seed=seed)
    env.resample_command_each_step = bool(resample_command)
    obs_dim = env.observation_space.shape[0]
    d_s = obs_dim + 1  # bias
    d_c = d_s * (d_s + 1) // 2

    actor = LinearGaussianActor(d_s, init_std=init_std)
    actor_opt = torch.optim.Adam([actor.theta], lr=alpha_a)
    critic = QuadraticCritic.zeros(d_c)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        z = phi_actor(obs)
        done = False
        trunc = False
        ep_return = 0.0
        steps = 0

        while not (done or trunc):
            # Torch sampling
            phi_t = torch.tensor(z, dtype=torch.float32)
            action_t, mean_t, log_std_t = actor.sample(phi_t, rng=torch_rng)
            action_np = action_t.detach().cpu().numpy()

            next_obs, reward, done, trunc, info = env.step(action_np)
            z_next = phi_actor(next_obs)

            v = critic.value(z)
            v_next = 0.0 if done else critic.value(z_next)
            delta = reward + gamma * v_next - v

            # Critic update (semi-gradient)
            critic.update(delta, z, alpha_c)

            # Actor update via autograd
            actor_opt.zero_grad()
            std_t = log_std_t.exp()
            dist = Normal(mean_t, std_t)
            log_prob = dist.log_prob(action_t).sum()
            loss = -(torch.tensor(delta, dtype=torch.float32) * log_prob)
            loss.backward()
            actor_opt.step()

            ep_return += reward
            steps += 1
            z = z_next

        returns.append(ep_return)
        if (ep + 1) % log_interval == 0:
            avg_ret = np.mean(returns[-log_interval:])
            print(f"[ep {ep+1}] avg_return({log_interval})={avg_ret:.3f} steps={steps}")
    return actor, critic, returns


def main():
    parser = argparse.ArgumentParser(description="Train canonical stride-level 3LP actor–critic")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha-a", type=float, default=1e-3)
    parser.add_argument("--alpha-c", type=float, default=5e-3)
    parser.add_argument("--init-std", type=float, default=0.5, help="Initial exploration std for each action dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--commands", type=float, nargs="+", default=[1.0], help="Forward speed commands (m/s)")
    parser.add_argument("--qv", type=float, default=0.0, help="Speed tracking weight")
    parser.add_argument("--q-state", type=float, nargs="+", default=None, help="Length-8 state cost diag")
    parser.add_argument("--r-act", type=float, nargs="+", default=None, help="Length-8 action cost diag")
    parser.add_argument("--u-limit", type=float, default=200.0)
    parser.add_argument("--reset-noise", type=float, default=0.01)
    parser.add_argument("--failure-threshold", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--resample-command", action="store_true", help="Resample command each stride to train transitions")
    args = parser.parse_args()

    env_kwargs = dict(
        commands=args.commands,
        q_x_diag=args.q_state,
        r_u_diag=args.r_act,
        q_v=args.qv,
        u_limit=args.u_limit,
        reset_noise_std=args.reset_noise,
        failure_threshold=args.failure_threshold,
        max_steps=args.max_steps,
        debug_log=False,
    )
    actor, critic, _ = train(
        episodes=args.episodes,
        gamma=args.gamma,
        alpha_a=args.alpha_a,
        alpha_c=args.alpha_c,
        init_std=args.init_std,
        seed=args.seed,
        log_interval=args.log_interval,
        env_kwargs=env_kwargs,
        resample_command=args.resample_command,
    )


if __name__ == "__main__":
    main()
