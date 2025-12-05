#!/usr/bin/env python
"""
Roll the high-rate env using its reference torques only (no policy deltas) to see
why/when it marks the nominal gait as fallen. Logs stance-frame bounds signals,
optionally plots them, and can visualize the rollout.

Usage:
  python scripts/debug_env_reference.py --steps 400 --plot runs/ref_bounds.png --visualize
  python scripts/debug_env_reference.py --v-cmd 1.0 --t-ds 0.1 --t-ss 0.6 --bounds 0.6 0.35 5 5
"""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_high_rate_3lp import ThreeLPHighRateEnv


def rollout_reference(env: ThreeLPHighRateEnv, steps: int, seed: int | None, v_cmd: float | None):
    options = {}
    if v_cmd is not None:
        options["v_cmd"] = float(v_cmd)
    obs, _ = env.reset(seed=seed, options=options if options else None)
    log = {
        "obs": [],
        "p_running": [],
        "tau_total": [],
        "fallen": [],
        "terminated": [],
        "phi": [],
        "phase": [],
        "state_world": [],
    }
    for _ in range(steps):
        # No delta action; use reference only.
        action = np.zeros(env.action_dim, dtype=np.float64)
        obs_next, reward, terminated, truncated, info = env.step(action)
        log["obs"].append(obs)
        log["p_running"].append(np.asarray(info.get("p_running", []), dtype=np.float64))
        log["tau_total"].append(np.asarray(info.get("tau_total", []), dtype=np.float64))
        log["fallen"].append(bool(info.get("fallen", False)))
        log["terminated"].append(bool(terminated))
        log["phi"].append(float(info.get("phi_stride", 0.0)))
        log["phase"].append(info.get("phase", ""))
        if "state_world" in info:
            log["state_world"].append(np.asarray(info["state_world"], dtype=np.float64))
        obs = obs_next
        if terminated or truncated:
            break
    return log


def plot_bounds(log: dict, dt: float, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional plotting
        print(f"[plot] matplotlib unavailable: {e}")
        return
    obs = np.asarray(log["obs"])
    if obs.size == 0:
        print("[plot] no obs logged")
        return
    s1x, s1y = obs[:, 0], obs[:, 1]
    ds1x, ds1y = obs[:, 4], obs[:, 5]
    t = np.arange(len(obs)) * dt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, s1x, label="s1x")
    axes[0].plot(t, s1y, label="s1y")
    axes[0].set_ylabel("stance pos")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, ds1x, label="ds1x")
    axes[1].plot(t, ds1y, label="ds1y")
    axes[1].set_ylabel("stance vel")
    axes[1].set_xlabel("time [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved to {out_path}")


def maybe_visualize(log: dict, env: ThreeLPHighRateEnv, loop: bool):
    try:
        import threelp  # type: ignore
    except Exception as e:  # pragma: no cover - GUI path
        print(f"[viz] threelp unavailable: {e}")
        return
    if not hasattr(threelp, "visualize_trajectory"):
        print("[viz] visualize_trajectory not available (build with PY_VISUALIZER=ON)")
        return
    if not log.get("state_world"):
        print("[viz] no state_world recorded")
        return
    states = []
    for q in log["state_world"]:
        st = threelp.ThreeLPState()
        st.q = [float(v) for v in q]
        states.append(st)
    try:
        params = env.sim.get_params() if hasattr(env, "sim") else threelp.ThreeLPParams.Adult()
        threelp.visualize_trajectory(
            states,
            env.t_ds,
            env.t_ss,
            params,
            fps=60.0,
            loop=loop,
            wait_for_close=loop,
        )
    except Exception as e:  # pragma: no cover - GUI path
        print(f"[viz] visualize_trajectory error: {e}")


def main():
    p = argparse.ArgumentParser(description="Debug reference-only rollout in ThreeLPHighRateEnv.")
    p.add_argument("--steps", type=int, default=400, help="Max steps to roll out.")
    p.add_argument("--v-cmd", type=float, default=None, help="Optional fixed command speed.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-ds", type=float, default=0.1)
    p.add_argument("--t-ss", type=float, default=0.6)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--bounds", type=float, nargs=4, default=None, help="fall_bounds override: s1x s1y ds1x ds1y")
    p.add_argument("--plot", type=str, default=None, help="Optional path to plot stance bounds.")
    p.add_argument("--visualize", action="store_true", help="Replay in Open3D if available.")
    p.add_argument("--loop", action="store_true", help="Loop visualization.")
    args = p.parse_args()

    env_kwargs = dict(
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        dt=args.dt,
        max_steps=args.max_env_steps,
    )
    if args.bounds:
        env_kwargs["fall_bounds"] = tuple(args.bounds)
    env = ThreeLPHighRateEnv(**env_kwargs)

    log = rollout_reference(env, args.steps, seed=args.seed, v_cmd=args.v_cmd)
    print(f"[ref] steps={len(log['obs'])} fallen={any(log['fallen'])}")

    if args.plot:
        plot_bounds(log, env.dt, Path(args.plot))
    if args.visualize:
        maybe_visualize(log, env, loop=args.loop)


if __name__ == "__main__":
    main()
