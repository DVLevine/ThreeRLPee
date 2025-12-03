"""
Minimal actor–critic loop for the canonical stride-level 3LP environment.

Implements the linear Gaussian actor and quadratic-feature critic from the
specification. One env.step = one stride; the environment already embeds the
closed-form stride map, so no numerical integration is used inside training.
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env_canonical_stride import ThreeLPCanonicalStrideEnv


def phi_actor(obs: np.ndarray) -> np.ndarray:
    """Actor features φ_a = [δx (8), c (1), bias]."""
    return np.concatenate([obs.astype(np.float64), np.array([1.0], dtype=np.float64)], axis=0)


def phi_critic(z: np.ndarray) -> np.ndarray:
    """Critic features φ_c = all unique quadratic monomials of z (upper triangle)."""
    feats = []
    d = z.shape[0]
    for i in range(d):
        for j in range(i, d):
            feats.append(z[i] * z[j])
    return np.asarray(feats, dtype=np.float64)


@dataclass
class LinearGaussianActor:
    theta: np.ndarray  # (8, d_s)
    sigma: np.ndarray  # (8,) std dev

    @classmethod
    def zeros(cls, d_s: int, init_std: float = 0.5):
        theta = np.zeros((8, d_s), dtype=np.float64)
        sigma = np.ones(8, dtype=np.float64) * init_std
        return cls(theta, sigma)

    def mean(self, z: np.ndarray) -> np.ndarray:
        return self.theta @ z

    def sample(self, z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        mu = self.mean(z)
        noise = rng.normal(scale=self.sigma, size=mu.shape)
        return mu + noise, mu

    def grad_log_pi(self, a: np.ndarray, mu: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Σ^{-1}(a-μ) φᵀ  with diagonal Σ
        inv_var = 1.0 / (self.sigma ** 2)
        diff = (a - mu) * inv_var
        return np.outer(diff, z)


@dataclass
class QuadraticCritic:
    w: np.ndarray  # (d_c,)

    @classmethod
    def zeros(cls, d_c: int):
        return cls(np.zeros(d_c, dtype=np.float64))

    def value(self, z: np.ndarray) -> float:
        phi = phi_critic(z)
        return float(self.w @ phi)

    def update(self, delta: float, z: np.ndarray, alpha_c: float):
        phi = phi_critic(z)
        self.w += alpha_c * delta * phi


def train(
    episodes: int,
    gamma: float,
    alpha_a: float,
    alpha_c: float,
    init_std: float,
    seed: int | None,
    log_interval: int,
    env_kwargs: dict,
):
    rng = np.random.default_rng(seed)
    env = ThreeLPCanonicalStrideEnv(**env_kwargs, seed=seed)

    obs_dim = env.observation_space.shape[0]
    d_s = obs_dim + 1  # bias added in phi_actor
    d_c = d_s * (d_s + 1) // 2

    actor = LinearGaussianActor.zeros(d_s, init_std=init_std)
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
            a, mu = actor.sample(z, rng)
            next_obs, reward, done, trunc, info = env.step(a)
            z_next = phi_actor(next_obs)

            v = critic.value(z)
            v_next = 0.0 if done else critic.value(z_next)
            delta = reward + gamma * v_next - v

            critic.update(delta, z, alpha_c)
            grad_log_pi = actor.grad_log_pi(a, mu, z)
            actor.theta += alpha_a * delta * grad_log_pi

            ep_return += reward
            steps += 1
            z = z_next

        returns.append(ep_return)
        if (ep + 1) % log_interval == 0:
            avg_ret = np.mean(returns[-log_interval:])
            print(f"[ep {ep+1}] avg_return({log_interval}) = {avg_ret:.3f}  last_steps={steps}")
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

    train(
        episodes=args.episodes,
        gamma=args.gamma,
        alpha_a=args.alpha_a,
        alpha_c=args.alpha_c,
        init_std=args.init_std,
        seed=args.seed,
        log_interval=args.log_interval,
        env_kwargs=env_kwargs,
    )


if __name__ == "__main__":
    main()
