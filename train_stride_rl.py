import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from env_canonical_stride import ThreeLPCanonicalStrideEnv
from stride_utils import JsonlLogger, make_run_dir, render_canonical_policy

# --- Feature Builders ---
def phi_actor(obs: np.ndarray) -> np.ndarray:
    """Actor features: [scaled_obs, 1.0]"""
    return np.concatenate([obs, [1.0]]).astype(np.float32)

def phi_critic(z: np.ndarray) -> np.ndarray:
    """Critic features: Quadratic monomials of z."""
    # z includes bias, so this naturally covers constant, linear, and quadratic terms
    n = len(z)
    feats = []
    for i in range(n):
        for j in range(i, n):
            feats.append(z[i] * z[j])
    return np.array(feats, dtype=np.float32)

# --- Models ---
class LinearGaussianActor(nn.Module):
    def __init__(self, input_dim, action_dim, init_std=0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(action_dim, input_dim))
        # Learnable log_std, initialized to log(init_std)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))

    def forward(self, x):
        # x: [input_dim]
        mean = torch.mv(self.theta, x)
        std = torch.exp(self.log_std)
        return mean, std

class QuadraticCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        return torch.dot(self.w, x)

# --- Training ---
def _save_checkpoint(run_dir: Path, tag: str, actor, critic, env_kwargs: dict, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "env_kwargs": env_kwargs,
        "args": vars(args),
    }
    torch.save(payload, run_dir / f"checkpoint_{tag}.pt")


def train(args):
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Env
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
        single_command_only=args.single_command_only,
    )
    env = ThreeLPCanonicalStrideEnv(**env_kwargs, seed=args.seed)

    # Dims
    obs_dim = env.observation_space.shape[0]
    actor_input_dim = obs_dim + 1
    
    # Critic dim: size of upper triangle of (actor_input_dim x actor_input_dim)
    critic_input_dim = actor_input_dim * (actor_input_dim + 1) // 2

    # Networks
    actor = LinearGaussianActor(actor_input_dim, env.action_space.shape[0], args.init_std)
    critic = QuadraticCritic(critic_input_dim)

    # Optimizers
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(prefix="stride_rl")
    print(f"[run] outputs will be written to {run_dir}")
    log_path = run_dir / "metrics.jsonl"
    logger = JsonlLogger(log_path) if args.log_metrics else None

    recent_returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        features_critic = []
        
        done = False
        while not done:
            # Prepare features
            z_actor = torch.tensor(phi_actor(obs))
            z_critic = torch.tensor(phi_critic(z_actor.numpy()))
            
            # Actor Step
            mean, std = actor(z_actor)
            dist = Normal(mean, std)
            action = dist.sample()
            
            # Env Step
            next_obs, reward, term, trunc, info = env.step(action.detach().numpy())
            done = term or trunc
            
            # Store
            log_probs.append(dist.log_prob(action).sum())
            values.append(critic(z_critic))
            features_critic.append(z_critic) # Store tensor for backward
            rewards.append(reward)
            
            obs = next_obs

        # --- Update (Episode End) ---
        
        # Calculate Returns (Monte Carlo)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns? (Often helps stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute Losses
        actor_loss = 0
        critic_loss = 0
        
        for log_p, v, G, z_c in zip(log_probs, values, returns, features_critic):
            advantage = G - v.item() # Detach baseline
            
            # Actor Gradients
            actor_loss += -log_p * advantage
            
            # Critic Gradients (MSE)
            # Re-calculate v graph here if needed, or accumulate gradients
            # Simple way: 
            v_pred = critic(z_c)
            critic_loss += F.mse_loss(v_pred, torch.tensor(G))

        # Backward Actor
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()

        # Backward Critic
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()

        # Logging
        ep_ret = sum(rewards)
        recent_returns.append(ep_ret)
        if logger:
            logger.log({
                "episode": ep,
                "return": float(ep_ret),
                "length": len(rewards),
            })

        if (ep+1) % args.log_interval == 0:
            avg = np.mean(recent_returns[-args.log_interval:])
            print(f"[Ep {ep+1}] Avg Ret: {avg:.2f} | Last Steps: {len(rewards)} | Sigma: {actor.log_std.exp().mean().item():.3f}")

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            _save_checkpoint(run_dir, f"ep{ep+1:05d}", actor, critic, env_kwargs, args)
        if ep == args.episodes - 1 and not args.skip_final_save:
            _save_checkpoint(run_dir, "final", actor, critic, env_kwargs, args)

        if args.viz_every > 0 and (ep + 1) % args.viz_every == 0:
            def _policy_fn(obs_np: np.ndarray) -> np.ndarray:
                z_actor = torch.tensor(phi_actor(obs_np))
                with torch.no_grad():
                    mean, _ = actor(z_actor)
                return mean.numpy()
            try:
                render_canonical_policy(
                    _policy_fn,
                    env_kwargs,
                    max_steps=args.viz_steps,
                    n_substeps=args.viz_substeps,
                    seed=args.seed,
                    loop=args.viz_loop,
                    log_prefix="viz",
                    backend=args.viz_backend,
                )
            except Exception as e:
                print(f"[viz] render error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--commands", type=float, nargs="+", default=[1.0])
    parser.add_argument("--qv", type=float, default=0.1)
    parser.add_argument("--q-state", type=float, nargs="+", default=None)
    parser.add_argument("--r-act", type=float, nargs="+", default=None)
    parser.add_argument("--u-limit", type=float, default=200.0)
    parser.add_argument("--reset-noise", type=float, default=0.01)
    parser.add_argument("--failure-threshold", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--viz-every", type=int, default=0, help="If >0, render a rollout every N episodes.")
    parser.add_argument("--viz-steps", type=int, default=50, help="Number of strides to show when visualizing.")
    parser.add_argument("--viz-substeps", type=int, default=120, help="Dense stride samples for visualization.")
    parser.add_argument("--viz-loop", action="store_true", help="Replay visualization window in a loop.")
    parser.add_argument("--viz-backend", type=str, choices=["python", "native", "auto"], default="python", help="Visualization backend for canonical strides.")
    parser.add_argument("--log-metrics", action="store_true", help="Write metrics to a JSONL log in the run dir.")
    parser.add_argument("--save-every", type=int, default=0, help="Checkpoint every N episodes (0=off).")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional run directory; defaults to runs/stride_rl_<timestamp>.")
    parser.add_argument("--skip-final-save", action="store_true", help="Skip saving the final checkpoint.")
    parser.add_argument("--single-command-only", action="store_true", help="Force a single command (no resampling) to test single-speed convergence.")
    args = parser.parse_args()
    
    train(args)
