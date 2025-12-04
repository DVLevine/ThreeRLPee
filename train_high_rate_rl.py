"""
Minimal actor–critic trainer for the high-rate 3LP environment.

Fixes:
 1) Phase features use sin/cos to respect stride periodicity.
 2) Terminal states do not bootstrap value estimates.
 3) Exploration noise scales with sqrt(dt).
 4) Rewards are scaled to avoid gradient explosions; TD error is clipped.
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from env_high_rate_3lp import ThreeLPHighRateEnv


def build_actor_features(env: ThreeLPHighRateEnv, obs: np.ndarray, v_nom: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Actor features z_k.
    Uses sin/cos embedding for phase to avoid discontinuity at stride wrap.
    """
    x = obs[:8]
    phi = float(obs[8])
    v_cmd = float(obs[9])
    x_ref = env.reference_state(phi) if hasattr(env, "reference_state") else np.zeros_like(x)
    e = x - x_ref
    sin_phi = np.sin(2 * np.pi * phi)
    cos_phi = np.cos(2 * np.pi * phi)
    # z dim: 8 (error) + 2 (phase) + 1 (command offset) = 11
    z = np.concatenate([e, [sin_phi, cos_phi, v_cmd - v_nom]], axis=0)
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
        alpha_p=0.05,
        p_decay=0.98,
        action_clip=10.0,
        alive_bonus=2.0,
    )
    v_nom = 0.5 * (env.v_cmd_range[0] + env.v_cmd_range[1])

    z_dim = 11
    critic_dim = 1 + z_dim + (z_dim * (z_dim + 1)) // 2

    W_a = np.zeros((env.action_dim, z_dim), dtype=np.float64)
    theta = np.zeros(critic_dim, dtype=np.float64)

    rng = np.random.default_rng(args.seed)

    log_path = Path(args.log_path) if args.log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting training: dt={args.dt}, sigma={args.sigma}, feature_dim={z_dim}")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep if args.seed is not None else None)
        done = False
        ep_ret = 0.0
        ep_len = 0

        z, _ = build_actor_features(env, obs, v_nom)

        while not done and ep_len < args.max_steps:
            mu = W_a @ z
            scaled_sigma = args.sigma * np.sqrt(args.dt)
            noise = rng.normal(scale=scaled_sigma, size=mu.shape)
            action = mu + noise

            obs_next, raw_reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            # Scale rewards to keep updates small and stable.
            reward = raw_reward * 0.01
            z_next, _ = build_actor_features(env, obs_next, v_nom)

            phi_c = critic_features(z)
            v_curr = float(theta @ phi_c)

            if terminated:
                v_next = 0.0
            else:
                phi_c_next = critic_features(z_next)
                v_next = float(theta @ phi_c_next)

            delta = reward + args.gamma * v_next - v_curr
            # Clip TD error to avoid shock updates.
            delta = np.clip(delta, -1.0, 1.0)

            theta += args.alpha_v * delta * phi_c

            effective_var = (scaled_sigma ** 2) if scaled_sigma > 0 else 1e-8
            grad_logpi = np.outer((action - mu) / effective_var, z)
            W_a += args.alpha_a * delta * grad_logpi

            if np.isnan(v_curr) or np.any(np.isnan(W_a)):
                print(f"NaN detected at episode {ep}, step {ep_len}")
                return

            obs = obs_next
            z = z_next
            ep_ret += raw_reward  # log unscaled reward for readability
            ep_len += 1
            done = terminated or truncated

        if (ep + 1) % 10 == 0:
            print(f"[ep {ep+1:04d}] return={ep_ret:.2f} len={ep_len}")
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ep+1},{ep_ret},{ep_len}\n")

    # Optionally persist weights
    if args.save_path:
        out = Path(args.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, W_a=W_a, theta=theta, config=vars(args))
        print(f"[save] wrote weights to {out}")


def _parse_args():
    p = argparse.ArgumentParser(description="Train high-rate 3LP actor-critic.")
    p.add_argument("--episodes", type=int, default=500, help="Number of episodes.")
    p.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode.")
    p.add_argument("--max-env-steps", type=int, default=5000, help="Env max steps before truncation.")
    p.add_argument("--dt", type=float, default=0.02, help="Control period Δt.")
    p.add_argument("--t-ds", type=float, default=0.1, help="Double support duration.")
    p.add_argument("--t-ss", type=float, default=0.6, help="Single support duration.")
    p.add_argument("--sigma", type=float, default=5.0, help="Base exploration std (scaled by sqrt(dt)).")
    p.add_argument("--alpha-a", type=float, default=1e-5, help="Actor learning rate.")
    p.add_argument("--alpha-v", type=float, default=1e-4, help="Critic learning rate.")
    p.add_argument("--gamma", type=float, default=0.995, help="Discount factor.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--log-path", type=str, default="logs/train_log.csv", help="Optional CSV log path.")
    p.add_argument("--save-path", type=str, default="weights/policy.npz", help="Optional .npz output for learned weights.")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
