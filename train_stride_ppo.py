import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from env_canonical_stride import ThreeLPCanonicalStrideEnv
from stride_utils import (
    JsonlLogger,
    make_run_dir,
    render_canonical_policy,
)
try:
    import threelp  # type: ignore
except Exception:
    threelp = None


def phi_actor(obs):
    # Linear features: [obs, 1.0]
    return np.concatenate([obs, [1.0]]).astype(np.float32)


def phi_critic(z):
    # Quadratic features
    n = len(z)
    feats = [z[i] * z[j] for i in range(n) for j in range(i, n)]
    return np.array(feats, dtype=np.float32)


class LinearActor(nn.Module):
    def __init__(self, in_dim, act_dim, init_std=0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(act_dim, in_dim))
        self.log_std = nn.Parameter(torch.ones(act_dim) * np.log(init_std))

    def forward(self, x):
        mean = torch.matmul(x, self.theta.T)
        return mean, self.log_std.exp()


class QuadraticCritic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return torch.matmul(x, self.w)


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
    # Setup
    env_kwargs = dict(
        commands=args.commands,
        q_v=args.qv,
        u_limit=args.u_limit,
        q_x_diag=args.q_state,
        r_u_diag=args.r_act,
        reset_noise_std=args.reset_noise,
        reset_noise_to_state=args.reset_noise_to_state,
        action_noise_std=args.action_noise_std,
        failure_threshold=args.failure_threshold,
        fail_penalty=args.fail_penalty,
        zmp_limit=args.zmp_limit,
        max_steps=args.max_steps,
        single_command_only=args.single_command_only,
    )
    env = ThreeLPCanonicalStrideEnv(**env_kwargs, seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Feature dims
    dim_actor = obs_dim + 1
    dim_critic = dim_actor * (dim_actor + 1) // 2

    actor = LinearActor(dim_actor, act_dim, args.init_std)
    critic = QuadraticCritic(dim_critic)

    # Optional warm-start from time-projection / DLQR gain.
    if args.tp_warm_start:
        if threelp is None:
            print("[tp] threelp module not available; skipping TP warm-start")
        else:
            cmd = args.tp_command if args.tp_command is not None else float(env.command_grid[0])
            try:
                tp = threelp.build_time_projection_controller(cmd, env.t_ds, env.t_ss, env.params, args.tp_use_uv)
                gain = np.array(tp.dlqr_gain(), dtype=np.float64)
                if gain.size == 0:
                    print("[tp] empty gain; skipping warm-start")
                else:
                    theta_np = np.zeros((act_dim, dim_actor), dtype=np.float64)
                    rows = min(act_dim, gain.shape[0])
                    cols = min(8, gain.shape[1]) if gain.ndim > 1 else 0
                    if cols > 0:
                        scale = (env.obs_scale[:cols] * env.u_limit).reshape(1, cols)
                        theta_np[:rows, :cols] = -gain[:rows, :cols] / scale
                    actor.theta.data = torch.as_tensor(theta_np, dtype=actor.theta.dtype)
                    actor.log_std.data[:] = np.log(args.tp_init_std if args.tp_init_std is not None else args.init_std)
                    print(f"[tp] warm-started actor from TP gain (cmd={cmd}, use_uv={args.tp_use_uv})")
            except Exception as e:
                print(f"[tp] warm-start failed: {e}")

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    steps_per_epoch = 2048

    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(prefix="stride_ppo")
    print(f"[run] outputs will be written to {run_dir}")
    log_path = run_dir / "metrics.jsonl"
    logger = JsonlLogger(log_path) if args.log_metrics else None

    for epoch in range(args.epochs):
        # --- Collect Batch ---
        buffer_obs, buffer_act, buffer_rew, buffer_val, buffer_logp = [], [], [], [], []
        buffer_feat_c = []
        buffer_done = []

        obs, _ = env.reset()
        ep_rew = 0
        ep_len = 0
        ep_infos = []

        for t in range(steps_per_epoch):
            # Features
            feat_a = torch.tensor(phi_actor(obs))
            feat_c = torch.tensor(phi_critic(feat_a.numpy()))

            # Action
            with torch.no_grad():
                mean, std = actor(feat_a)
                dist = Normal(mean, std)
                action = dist.sample()
                logp = dist.log_prob(action).sum()
                val = critic(feat_c)

            next_obs, rew, term, trunc, info = env.step(action.numpy())

            # Store
            buffer_obs.append(feat_a)
            buffer_act.append(action)
            buffer_rew.append(rew)
            buffer_val.append(val)
            buffer_logp.append(logp)
            buffer_feat_c.append(feat_c)
            buffer_done.append(float(term or trunc))

            obs = next_obs
            ep_rew += rew
            ep_len += 1

            if term or trunc:
                ep_infos.append({'rew': ep_rew, 'len': ep_len})
                obs, _ = env.reset()
                ep_rew = 0
                ep_len = 0

        # --- GAE Calculation ---
        # Bootstrap last value
        feat_a = torch.tensor(phi_actor(obs))
        feat_c = torch.tensor(phi_critic(feat_a.numpy()))
        with torch.no_grad():
            last_val = critic(feat_c)

        buffer_adv = np.zeros(steps_per_epoch, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(steps_per_epoch)):
            if t == steps_per_epoch - 1:
                next_val = last_val
            else:
                next_val = buffer_val[t + 1]
            nonterminal = 1.0 - buffer_done[t]
            delta = buffer_rew[t] + args.gamma * next_val * nonterminal - buffer_val[t]
            last_gae = delta + args.gamma * 0.95 * nonterminal * last_gae  # GAE-Lambda with done masking
            buffer_adv[t] = last_gae

        buffer_ret = torch.tensor(buffer_adv) + torch.stack(buffer_val)
        buffer_adv = torch.tensor(buffer_adv)
        buffer_adv = (buffer_adv - buffer_adv.mean()) / (buffer_adv.std() + 1e-8)

        # --- PPO Update ---
        b_obs = torch.stack(buffer_obs)
        b_act = torch.stack(buffer_act)
        b_logp = torch.stack(buffer_logp)
        b_feat_c = torch.stack(buffer_feat_c)

        for _ in range(10):  # Update epochs
            # Actor Loss
            mean, std = actor(b_obs)
            dist = Normal(mean, std)
            new_logp = dist.log_prob(b_act).sum(axis=1)
            entropy = dist.entropy().sum(axis=1)
            ratio = torch.exp(new_logp - b_logp)

            surr1 = ratio * buffer_adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * buffer_adv
            loss_pi = -(torch.min(surr1, surr2) + args.ent_coef * entropy).mean()

            opt_actor.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            opt_actor.step()

            # Critic Loss
            v_pred = critic(b_feat_c)
            loss_v = ((v_pred - buffer_ret) ** 2).mean()

            opt_critic.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            opt_critic.step()

        # Log
        avg_ret = np.mean([i['rew'] for i in ep_infos]) if ep_infos else 0.0
        avg_len = np.mean([i['len'] for i in ep_infos]) if ep_infos else 0.0
        print(f"Epoch {epoch}: Avg Ret {avg_ret:.2f} | Avg Len {avg_len:.1f}")

        if logger:
            logger.log({
                "epoch": epoch,
                "avg_return": float(avg_ret),
                "avg_len": float(avg_len),
            })

        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(run_dir, f"epoch{epoch:04d}", actor, critic, env_kwargs, args)
        if epoch == args.epochs - 1 and not args.skip_final_save:
            _save_checkpoint(run_dir, "final", actor, critic, env_kwargs, args)

        if args.viz_every > 0 and (epoch + 1) % args.viz_every == 0:
            def _policy_fn(obs_np: np.ndarray) -> np.ndarray:
                feat_a = torch.tensor(phi_actor(obs_np))
                with torch.no_grad():
                    mean, _ = actor(feat_a)
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--commands", type=float, nargs="+", default=[1.2])
    parser.add_argument("--qv", type=float, default=0.1)
    parser.add_argument("--q-state", type=float, nargs="+", default=None, help="Length-8 state cost diag")
    parser.add_argument("--r-act", type=float, nargs="+", default=None, help="Length-8 action cost diag")
    parser.add_argument("--u-limit", type=float, default=200.0)
    parser.add_argument("--reset-noise", type=float, default=0.01)
    parser.add_argument("--reset-noise-to-state", action="store_true", help="Apply reset noise to policy state (delta_x). Off means start exactly at x_ref.")
    parser.add_argument("--action-noise-std", type=float, default=0.0, help="Add Gaussian exploration noise to actions inside the env.")
    parser.add_argument("--failure-threshold", type=float, default=5.0)
    parser.add_argument("--fail-penalty", type=float, default=50.0)
    parser.add_argument("--zmp-limit", type=float, default=0.25, help="ZMP/COP limit in meters; raise to relax early ZMP terminations.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--viz-every", type=int, default=0, help="If >0, render a rollout every N epochs.")
    parser.add_argument("--viz-steps", type=int, default=50, help="Number of strides to show when visualizing.")
    parser.add_argument("--viz-substeps", type=int, default=120, help="Dense stride samples for visualization.")
    parser.add_argument("--viz-loop", action="store_true", help="Replay visualization window in a loop.")
    parser.add_argument("--viz-backend", type=str, choices=["python", "native", "auto"], default="python", help="Visualization backend for canonical strides.")
    parser.add_argument("--log-metrics", action="store_true", help="Write metrics to a JSONL log in the run dir.")
    parser.add_argument("--save-every", type=int, default=0, help="Checkpoint every N epochs (0=off).")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional run directory; defaults to runs/stride_ppo_<timestamp>.")
    parser.add_argument("--skip-final-save", action="store_true", help="Skip saving the final checkpoint.")
    parser.add_argument("--single-command-only", action="store_true", help="Force single command (no resampling) for convergence debugging.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy bonus coefficient for the policy update.")
    parser.add_argument("--tp-warm-start", action="store_true", help="Initialize actor weights from time-projection DLQR gain.")
    parser.add_argument("--tp-use-uv", action="store_true", default=True, help="Use U+V torques in TP warm-start (default), otherwise U-only.")
    parser.add_argument("--tp-command", type=float, default=None, help="Command speed to build TP controller; defaults to first env command (TP controller tuned for ~1.2 m/s).")
    parser.add_argument("--tp-init-std", type=float, default=None, help="Override init std after TP warm-start; defaults to init_std.")
    parser.add_argument("--tp-fit-actor", action="store_true", help="Fit actor weights to TP projections via least squares on sampled delta_x.")
    parser.add_argument("--tp-fit-samples", type=int, default=512, help="Number of TP samples for fitting actor.")
    parser.add_argument("--tp-fit-noise", type=float, default=0.02, help="Std of delta_x sampling for TP fit.")
    args = parser.parse_args()

    train(args)
