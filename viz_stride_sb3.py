import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from env_threelp_stride_sac import ThreeLPStrideEnv

try:
    import threelp  # type: ignore
except Exception:
    threelp = None


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_env_from_config(
    cfg: Dict[str, Any],
    seed: Optional[int],
    v_cmd_override: Optional[float] = None,
    reset_noise_override: Optional[float] = None,
) -> ThreeLPStrideEnv:
    """Build stride env using fields saved by the trainer config, with optional overrides."""
    v_cfg = cfg.get("v_cmd", None)
    v_range = (v_cfg, v_cfg) if v_cfg is not None else (cfg.get("v_cmd_min", 0.6), cfg.get("v_cmd_max", 1.4))
    if v_cmd_override is not None:
        v_range = (v_cmd_override, v_cmd_override)

    kwargs = {
        "t_ds": cfg.get("t_ds", 0.1),
        "t_ss": cfg.get("t_ss", 0.6),
        "inner_dt": cfg.get("inner_dt", 0.005),
        "max_strides": cfg.get("max_strides", 200),
        "action_scale": cfg.get("action_scale", 40.0),
        "alive_bonus": cfg.get("alive_bonus", 5.0),
        "q_e_diag": tuple(cfg.get("q_e_diag", (20.0, 20.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0))),
        "q_v": cfg.get("q_v", 2.0),
        "r_action": cfg.get("r_action", 1e-3),
        "terminal_penalty": cfg.get("terminal_penalty", 50.0),
        "fall_bounds": tuple(cfg.get("fall_bounds", (1.0, 0.5, 10.0, 10.0))),
        "v_cmd_range": v_range,
        "reset_noise_std": cfg.get("reset_noise_std", 0.0),
        "ref_substeps": cfg.get("ref_substeps", None),
        "obs_clip": cfg.get("obs_clip", 1e3),
        "seed": seed,
    }
    if reset_noise_override is not None:
        kwargs["reset_noise_std"] = reset_noise_override
    env = ThreeLPStrideEnv(**kwargs)
    # Optional warm cache to avoid first-episode cost spikes.
    precache = cfg.get("precache_cmds", None)
    if precache:
        for v in precache:
            try:
                env._get_reference(float(v))
            except Exception as exc:  # pragma: no cover - visualization helper
                print(f"[viz] precache failed for v_cmd={v}: {exc}")
    return env


def _to_state_obj(q: np.ndarray):
    s = threelp.ThreeLPState()
    s.q = [float(v) for v in q.reshape(-1)]
    return s


def maybe_visualize(states: List[np.ndarray], env: ThreeLPStrideEnv, loop: bool) -> None:
    if not states:
        print("[viz] no states captured; nothing to visualize")
        return
    if threelp is None:
        print("[viz] threelp module not available; skipping visualization")
        return
    if not hasattr(threelp, "visualize_trajectory"):
        print("[viz] visualize_trajectory not available (build pybind with visualizer)")
        return
    try:
        params = env.sim.get_params() if hasattr(env.sim, "get_params") else getattr(env.sim, "params", None)
    except Exception:
        params = None
    if params is None:
        print("[viz] could not fetch sim params; skipping visualization")
        return

    try:
        state_objs = [_to_state_obj(np.asarray(q, dtype=np.float64)) for q in states]
        threelp.visualize_trajectory(
            state_objs,
            t_ds=env.t_ds,
            t_ss=env.t_ss,
            params=params,
            fps=60.0,
            loop=loop,
            wait_for_close=True,
        )
    except Exception as exc:  # pragma: no cover - visualization helper
        print(f"[viz] visualize_trajectory error: {exc}")


def rollout(model: SAC, vec_env: DummyVecEnv, steps: int, deterministic: bool) -> tuple[list[np.ndarray], list[float], list[dict]]:
    env = vec_env.envs[0]
    env.capture_trajectory = True

    obs = vec_env.reset()
    states: List[np.ndarray] = []
    rewards: List[float] = []
    infos: List[dict] = []

    # Starting state after reset
    if hasattr(env, "state_q"):
        states.append(np.asarray(env.state_q, dtype=np.float64))

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rew, done, info = vec_env.step(action)

        info0 = info[0] if isinstance(info, (list, tuple)) else info
        traj = info0.get("trajectory")
        if traj:
            states.extend(np.asarray(traj, dtype=np.float64))

        rewards.append(float(rew[0]))
        infos.append(info0)

        if done[0]:
            break

    return states, rewards, infos


def parse_args():
    p = argparse.ArgumentParser(description="Visualize SAC stride policy rollouts.")
    p.add_argument("--run-dir", type=str, default=None, help="Training run directory containing config.json and sac_stride.zip.")
    p.add_argument("--model", type=str, default=None, help="Path to SAC model .zip (default: run-dir/sac_stride.zip).")
    p.add_argument("--config", type=str, default=None, help="Path to config.json saved by the trainer (default: run-dir/config.json).")
    p.add_argument("--steps", type=int, default=40, help="Number of strides to roll out (overridden by --seconds).")
    p.add_argument("--seconds", type=float, default=None, help="If set, roll out for this many seconds (converted to strides).")
    p.add_argument("--episodes", type=int, default=1, help="Number of independent rollouts to run.")
    p.add_argument("--seed", type=int, default=None, help="Base seed for env reset (each episode offsets by +episode index).")
    p.add_argument("--v-cmd", type=float, default=None, help="Override commanded speed for rollout.")
    p.add_argument("--reset-noise", type=float, default=None, help="Override reset noise std for rollout.")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic SAC actions.")
    p.add_argument("--visualize", action="store_true", help="Replay rollout in Open3D if available.")
    p.add_argument("--visualize-episode", type=int, default=0, help="Which episode index to visualize (0-based).")
    p.add_argument("--loop", action="store_true", help="Loop the visualization playback.")
    p.add_argument("--save-trajectory", type=str, default=None, help="Optional path prefix to save captured states as .npy")
    return p.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    model_path = Path(args.model) if args.model else (run_dir / "sac_stride.zip" if run_dir else None)
    cfg_path = Path(args.config) if args.config else (run_dir / "config.json" if run_dir else None)

    if model_path is None or cfg_path is None:
        raise SystemExit("Provide --run-dir or both --model and --config.")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    env = build_env_from_config(cfg, seed=args.seed, v_cmd_override=args.v_cmd, reset_noise_override=args.reset_noise)
    vec_env = DummyVecEnv([lambda: env])
    model = SAC.load(model_path, env=vec_env)

    # Determine stride count from seconds if requested.
    steps = args.steps
    if args.seconds is not None and env.stride_time > 0:
        steps = max(1, int(np.ceil(args.seconds / env.stride_time)))

    all_states: List[List[np.ndarray]] = []
    all_rewards: List[List[float]] = []
    all_infos: List[List[dict]] = []

    for ep in range(max(1, args.episodes)):
        seed_ep = args.seed + ep if args.seed is not None else None
        env.capture_trajectory = True
        if seed_ep is not None:
            try:
                vec_env.seed(seed_ep)
            except Exception:
                # Older SB3 DummyVecEnv has no seed(); fall back to env RNG.
                try:
                    env.rng = np.random.default_rng(seed_ep)
                except Exception:
                    pass
        obs = vec_env.reset()
        states, rewards, infos = rollout(model, vec_env, steps=steps, deterministic=args.deterministic)
        all_states.append(states)
        all_rewards.append(rewards)
        all_infos.append(infos)
        print(f"[rollout ep={ep}] strides={len(rewards)}  total_reward={sum(rewards):.3f}  fallen={infos[-1].get('fallen') if infos else None}")

        if args.save_trajectory:
            path = Path(args.save_trajectory)
            path_ep = path if args.episodes == 1 else path.with_stem(f"{path.stem}_ep{ep}")
            np.save(path_ep, np.asarray(states, dtype=np.float64))
            print(f"[rollout ep={ep}] saved trajectory to {path_ep}")

    if args.visualize:
        ep_idx = max(0, min(args.visualize_episode, len(all_states) - 1))
        maybe_visualize(all_states[ep_idx], env, loop=args.loop)


if __name__ == "__main__":
    main()
