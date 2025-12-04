"""
Minimal actor–critic trainer for the high-rate 3LP environment.

Implements the linear Gaussian actor and quadratic-monomial critic described in
the "High-Rate 3LP-Based RL" specification. Training is on-policy and purely
numpy-based to keep the update logic transparent.
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from env_high_rate_3lp import ThreeLPHighRateEnv


def build_actor_features(env: ThreeLPHighRateEnv, obs: np.ndarray, v_nom: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Actor features z_k = [e_k, phi-0.5, v_cmd-v_nom], where e_k is the reduced
    state error against the reference gait at the current phase.
    """
    x = obs[:8]
    phi = float(obs[8])
    v_cmd = float(obs[9])
    if hasattr(env, "reference_state"):
        x_ref = env.reference_state(phi)  # type: ignore[operator]
    else:
        x_ref = np.zeros_like(x)
    e = x - x_ref
    z = np.concatenate([e, [phi - 0.5, v_cmd - v_nom]], axis=0)
    return z.astype(np.float64), x_ref


def critic_features(z: np.ndarray) -> np.ndarray:
    """
    Quadratic monomial critic basis: [1, z, vec_upper(z z^T)].
    Vectorized for performance.
    """
    z = np.asarray(z, dtype=np.float64)
    quad_terms = np.outer(z, z)[np.triu_indices(len(z))]
    return np.concatenate([[1.0], z, quad_terms])


def train(args):
    env = ThreeLPHighRateEnv(
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        dt=args.dt,
        max_steps=args.max_env_steps,
    )
    v_nom = 0.5 * (env.v_cmd_range[0] + env.v_cmd_range[1])

    W_a = np.zeros((env.action_dim, 10), dtype=np.float64)
    theta = np.zeros(66, dtype=np.float64)  # 1 + 10 + 55

    rng = np.random.default_rng(args.seed)

    log_path = Path(args.log_path) if args.log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep if args.seed is not None else None)
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done and ep_len < args.max_steps:
            z, _ = build_actor_features(env, obs, v_nom)
            mu = W_a @ z
            scaled_sigma = args.sigma * np.sqrt(args.dt)
            noise = rng.normal(scale=scaled_sigma, size=mu.shape)
            action = mu + noise

            obs_next, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            z_next, _ = build_actor_features(env, obs_next, v_nom)

            phi_c = critic_features(z)
            phi_c_next = critic_features(z_next)
            delta = reward + args.gamma * float(theta @ phi_c_next) - float(theta @ phi_c)

            # Critic update
            theta += args.alpha_v * delta * phi_c

            # Actor update (score function for diagonal Gaussian with fixed σ)
            grad_logpi = np.outer((action - mu) / (args.sigma ** 2), z)
            W_a += args.alpha_a * delta * grad_logpi

            obs = obs_next
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated

        print(f"[ep {ep:04d}] return={ep_ret:.2f} len={ep_len}")
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ep},{ep_ret},{ep_len}\n")

    # Optionally persist weights
    if args.save_path:
        out = Path(args.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, W_a=W_a, theta=theta, config=vars(args))
        print(f"[save] wrote weights to {out}")


def _parse_args():
    p = argparse.ArgumentParser(description="Train high-rate 3LP actor-critic (linear Gaussian actor + quadratic critic).")
    p.add_argument("--episodes", type=int, default=200, help="Number of episodes.")
    p.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode.")
    p.add_argument("--max-env-steps", type=int, default=5000, help="Env max steps before truncation.")
    p.add_argument("--dt", type=float, default=0.02, help="Control period Δt.")
    p.add_argument("--t-ds", type=float, default=0.1, help="Double support duration.")
    p.add_argument("--t-ss", type=float, default=0.6, help="Single support duration.")
    p.add_argument("--sigma", type=float, default=1.0, help="Base stddev for Gaussian exploration; scaled by sqrt(dt).")
    p.add_argument("--alpha-a", type=float, default=1e-4, help="Actor learning rate.")
    p.add_argument("--alpha-v", type=float, default=5e-4, help="Critic learning rate.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--log-path", type=str, default="", help="Optional CSV log path (ep,return,len).")
    p.add_argument("--save-path", type=str, default="", help="Optional .npz output for learned weights.")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
