import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from env_canonical_stride import ThreeLPCanonicalStrideEnv
from stride_utils import JsonlLogger, make_run_dir, render_canonical_policy, lift_canonical_state
try:
    import threelp  # type: ignore
except Exception:
    threelp = None

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
        reset_noise_to_state=args.reset_noise_to_state,
        action_noise_std=args.action_noise_std,
        failure_threshold=args.failure_threshold,
        fail_penalty=args.fail_penalty,
        zmp_limit=args.zmp_limit,
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

    # Optional warm-start from time-projection / DLQR gain or regression fit.
    if args.tp_warm_start or args.tp_fit_actor:
        if threelp is None:
            print("[tp] threelp module not available; skipping TP warm-start/fit")
        else:
            cmd = args.tp_command if args.tp_command is not None else float(env.command_grid[0])
            try:
                tp = threelp.build_time_projection_controller(cmd, env.t_ds, env.t_ss, env.params, args.tp_use_uv)
                if args.tp_warm_start:
                    gain = np.array(tp.dlqr_gain(), dtype=np.float64)
                    if gain.size == 0:
                        print("[tp] empty gain; skipping warm-start")
                    else:
                        theta_np = np.zeros((env.action_space.shape[0], actor_input_dim), dtype=np.float64)
                        rows = min(theta_np.shape[0], gain.shape[0])
                        cols = min(8, gain.shape[1]) if gain.ndim > 1 else 0
                        if cols > 0:
                            scale = (env.obs_scale[:cols] * env.u_limit).reshape(1, cols)
                            theta_np[:rows, :cols] = -gain[:rows, :cols] / scale
                        actor.theta.data = torch.as_tensor(theta_np, dtype=actor.theta.dtype)
                        actor.log_std.data[:] = np.log(args.tp_init_std if args.tp_init_std is not None else args.init_std)
                        print(f"[tp] warm-started actor from TP gain (cmd={cmd}, use_uv={args.tp_use_uv})")
                if args.tp_fit_actor:
                    n = args.tp_fit_samples
                    rng = np.random.default_rng(args.seed)
                    base = env.maps[0]
                    z_list = []
                    y_list = []
                    for i in range(n):
                        leg = 1 if (i % 2 == 0) else -1
                        delta = rng.normal(0, args.tp_fit_noise, size=8)
                        x_abs = base.x_ref + delta
                        q = lift_canonical_state(x_abs)
                        a_phys = tp.project("ds", 0.0, q, leg, args.u_limit)
                        a_norm = np.clip(a_phys / env.u_limit, -1.0, 1.0)
                        obs_vec = np.concatenate([delta * env.obs_scale, [base.command]]).astype(np.float32)
                        z = phi_actor(obs_vec)
                        z_list.append(z)
                        y_list.append(a_norm)
                    Z = np.stack(z_list, axis=0)  # [n, input_dim]
                    Y = np.stack(y_list, axis=0)  # [n, act_dim]
                    theta_ls, *_ = np.linalg.lstsq(Z, Y, rcond=None)
                    theta_np = theta_ls.T
                    actor.theta.data = torch.as_tensor(theta_np, dtype=actor.theta.dtype)
                    actor.log_std.data[:] = np.log(args.tp_init_std if args.tp_init_std is not None else args.init_std)
                    print(f\"[tp] fitted actor to {n} TP samples (noise={args.tp_fit_noise}, cmd={cmd})\")
            except Exception as e:
                print(f\"[tp] warm-start/fit failed: {e}\")

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
        entropies = []
        
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
            entropies.append(dist.entropy().sum())
            
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
        
        for log_p, v, G, z_c, ent in zip(log_probs, values, returns, features_critic, entropies):
            advantage = G - v.item() # Detach baseline
            
            # Actor Gradients
            actor_loss += -log_p * advantage - args.ent_coef * ent
            
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
    parser.add_argument("--commands", type=float, nargs="+", default=[1.2])
    parser.add_argument("--qv", type=float, default=0.1)
    parser.add_argument("--q-state", type=float, nargs="+", default=None)
    parser.add_argument("--r-act", type=float, nargs="+", default=None)
    parser.add_argument("--u-limit", type=float, default=200.0)
    parser.add_argument("--reset-noise", type=float, default=0.01)
    parser.add_argument("--reset-noise-to-state", action="store_true", help="Apply reset noise to policy state (delta_x). Off means start exactly at x_ref.")
    parser.add_argument("--action-noise-std", type=float, default=0.0, help="Add Gaussian exploration noise to actions inside the env.")
    parser.add_argument("--failure-threshold", type=float, default=5.0)
    parser.add_argument("--fail-penalty", type=float, default=50.0)
    parser.add_argument("--zmp-limit", type=float, default=0.25, help="ZMP/COP limit in meters; raise to relax early ZMP terminations.")
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
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy bonus coefficient for policy exploration.")
    parser.add_argument("--tp-warm-start", action="store_true", help="Initialize actor weights from time-projection DLQR gain.")
    parser.add_argument("--tp-use-uv", action="store_true", help="Use U+V torques in TP warm-start (default), otherwise U-only.", default=True)
    parser.add_argument("--tp-command", type=float, default=None, help="Command speed to build TP controller; defaults to first env command (recommended 1.2 for TP).")
    parser.add_argument("--tp-init-std", type=float, default=None, help="Override init std after TP warm-start; defaults to init_std.")
    parser.add_argument("--tp-fit-actor", action="store_true", help="Fit actor weights to TP projections via least squares on sampled delta_x.")
    parser.add_argument("--tp-fit-samples", type=int, default=512, help="Number of TP samples for fitting actor.")
    parser.add_argument("--tp-fit-noise", type=float, default=0.02, help="Std of delta_x sampling for TP fit.")
    args = parser.parse_args()
    
    train(args)
