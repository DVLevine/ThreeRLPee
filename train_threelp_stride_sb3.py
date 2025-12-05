import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env_threelp_stride_sac import ThreeLPStrideEnv


def _build_env(args, cache, seed: Optional[int] = None) -> ThreeLPStrideEnv:
    """Factory to build a stride-level env with shared reference cache."""
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
        reference_cache=cache,
        obs_clip=args.obs_clip,
        seed=seed,
    )
    return env


def _precache(env: ThreeLPStrideEnv, cmds: Iterable[float]) -> None:
    """Optional: warm up reference cache so first episodes avoid the cost."""
    for v in cmds:
        try:
            env._get_reference(float(v))
        except Exception as exc:
            print(f"[precache] failed for v_cmd={v}: {exc}")


def train(args):
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"stride_sb3_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb"
    best_model_dir = run_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    shared_cache = {}

    def make_env(seed_offset: int = 0):
        def thunk():
            env = _build_env(args, shared_cache, seed=args.seed + seed_offset if args.seed is not None else None)
            if args.precache_cmds:
                _precache(env, args.precache_cmds)
            return env
        return thunk

    train_env = DummyVecEnv([make_env(0)])
    # Record per-episode rewards/lengths for plotting (monitor.csv).
    train_env = VecMonitor(train_env, filename=str(run_dir / "monitor"))
    eval_env = DummyVecEnv([make_env(10_000)])  # decorrelate seeds a bit

    policy_kwargs = dict(net_arch=dict(pi=[args.hidden_dim, args.hidden_dim], qf=[args.hidden_dim, args.hidden_dim]))

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        gamma=args.gamma,
        tau=args.tau,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
    )

    callbacks = []
    if args.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                best_model_save_path=str(best_model_dir),
                log_path=str(run_dir / "eval"),
                verbose=1,
            )
        )

    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks if callbacks else None, progress_bar=True)

    save_path = Path(args.save_path) if args.save_path else run_dir / "sac_stride.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"[done] saved model to {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Stable-Baselines3 SAC trainer for ThreeLP stride env.")

    # Env
    p.add_argument("--t-ds", type=float, default=0.1)
    p.add_argument("--t-ss", type=float, default=0.6)
    p.add_argument("--inner-dt", type=float, default=0.005)
    p.add_argument("--max-strides", type=int, default=200)
    p.add_argument("--action-scale", type=float, default=40.0)
    p.add_argument("--alive-bonus", type=float, default=5.0)
    p.add_argument("--q-e-diag", type=float, nargs=8, default=(20.0, 20.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0))
    p.add_argument("--q-v", type=float, default=2.0)
    p.add_argument("--r-action", type=float, default=1e-3)
    p.add_argument("--terminal-penalty", type=float, default=50.0)
    p.add_argument("--fall-bounds", type=float, nargs=4, default=(1.0, 0.5, 10.0, 10.0))
    p.add_argument("--reset-noise-std", type=float, default=0.01)
    p.add_argument("--obs-clip", type=float, default=1e3)
    p.add_argument("--v-cmd", type=float, default=None, help="Fix command speed if provided.")
    p.add_argument("--v-cmd-min", type=float, default=0.6)
    p.add_argument("--v-cmd-max", type=float, default=1.4)
    p.add_argument("--ref-substeps", type=int, default=None)
    p.add_argument("--precache-cmds", type=float, nargs="*", default=None)

    # SAC
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)

    # Logging / eval
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
