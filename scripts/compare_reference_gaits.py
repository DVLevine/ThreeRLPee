#!/usr/bin/env python
"""
Compare the env's reference-driven rollout (zero action, p_ref only) with the
open-loop canonical reference stride. Plots pelvis/foot XY paths and optionally
tries to visualize both; if a dual-trajectory visualizer is exposed in pybind,
it will use that, otherwise falls back to single-trajectory playback.

Usage:
  python scripts/compare_reference_gaits.py --steps 400 --strides 4 --plot runs/ref_compare.png --visualize
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_high_rate_3lp import ThreeLPHighRateEnv
from scripts.visualize_reference_stride import sample_reference_world  # reuse stride sampler


def rollout_env_reference(env: ThreeLPHighRateEnv, steps: int, seed: int | None, v_cmd: float | None):
    options = {}
    if v_cmd is not None:
        options["v_cmd"] = float(v_cmd)
    obs, _ = env.reset(seed=seed, options=options if options else None)
    world_states = []
    for _ in range(steps):
        action = np.zeros(env.action_dim, dtype=np.float64)
        obs_next, reward, terminated, truncated, info = env.step(action)
        if "state_world" in info:
            world_states.append(np.asarray(info["state_world"], dtype=np.float64))
        obs = obs_next
        if terminated or truncated:
            break
    return world_states


def _extract_xy(states: list[np.ndarray]):
    arr = np.asarray(states)
    pelvis = arr[:, 2:4]
    swing = arr[:, 0:2]
    stance = arr[:, 4:6]
    return pelvis, swing, stance


def plot_paths(env_states: list[np.ndarray], ref_states: list[np.ndarray], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional plotting
        print(f"[plot] matplotlib unavailable: {e}")
        return
    p_env, sw_env, st_env = _extract_xy(env_states)
    p_ref, sw_ref, st_ref = _extract_xy(ref_states)
    plt.figure(figsize=(8, 6))
    plt.plot(p_env[:, 0], p_env[:, 1], label="env pelvis", color="C0")
    plt.plot(sw_env[:, 0], sw_env[:, 1], "--", label="env swing", color="C0", alpha=0.6)
    plt.plot(st_env[:, 0], st_env[:, 1], ":", label="env stance", color="C0", alpha=0.4)
    plt.plot(p_ref[:, 0], p_ref[:, 1], label="ref pelvis", color="C1")
    plt.plot(sw_ref[:, 0], sw_ref[:, 1], "--", label="ref swing", color="C1", alpha=0.6)
    plt.plot(st_ref[:, 0], st_ref[:, 1], ":", label="ref stance", color="C1", alpha=0.4)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved to {out_path}")


def _maybe_visualize_dual(env_states: list[np.ndarray], ref_states: list[np.ndarray], env: ThreeLPHighRateEnv, loop: bool):
    try:
        import threelp  # type: ignore
    except Exception as e:  # pragma: no cover - GUI path
        print(f"[viz] threelp unavailable: {e}")
        return
    params = env.sim.get_params() if hasattr(env, "sim") else threelp.ThreeLPParams.Adult()

    # Prefer dual-trajectory if exposed.
    if hasattr(threelp, "visualize_dual_trajectory"):
        try:
            policy_states = []
            ref_states_obj = []
            for q in env_states:
                s = threelp.ThreeLPState()
                s.q = [float(v) for v in q]
                policy_states.append(s)
            for q in ref_states:
                s = threelp.ThreeLPState()
                s.q = [float(v) for v in q]
                ref_states_obj.append(s)
            threelp.visualize_dual_trajectory(
                policy_states,
                ref_states_obj,
                params,
                fps=60.0,
                loop=loop,
                wait_for_close=loop,
            )
            return
        except Exception as e:
            print(f"[viz] visualize_dual_trajectory error: {e}")
            # fall through to single playback

    if not hasattr(threelp, "visualize_trajectory"):
        print("[viz] visualize_trajectory not available (build with PY_VISUALIZER=ON)")
        return
    if not env_states:
        print("[viz] no env states to show")
        return
    try:
        policy_objs = []
        for q in env_states:
            s = threelp.ThreeLPState()
            s.q = [float(v) for v in q]
            policy_objs.append(s)
        threelp.visualize_trajectory(
            policy_objs,
            env.t_ds,
            env.t_ss,
            params,
            fps=60.0,
            loop=loop,
            wait_for_close=loop,
        )
    except Exception as e:
        print(f"[viz] visualize_trajectory error: {e}")


def main():
    p = argparse.ArgumentParser(description="Compare env reference rollout vs canonical reference stride.")
    p.add_argument("--steps", type=int, default=400, help="Max dt steps for env rollout.")
    p.add_argument("--strides", type=int, default=4, help="Strides to sample for canonical reference.")
    p.add_argument("--substeps", type=int, default=150, help="Samples per stride for canonical reference.")
    p.add_argument("--v-cmd", type=float, default=None, help="Optional fixed command speed.")
    p.add_argument("--t-ds", type=float, default=0.1)
    p.add_argument("--t-ss", type=float, default=0.6)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--plot", type=str, default=None, help="Optional PNG path for XY overlay.")
    p.add_argument("--visualize", action="store_true", help="Visualize trajectories (dual if available).")
    p.add_argument("--loop", action="store_true", help="Loop visualization playback.")
    args = p.parse_args()

    env_kwargs = dict(t_ds=args.t_ds, t_ss=args.t_ss, dt=args.dt, max_steps=args.max_env_steps)
    env = ThreeLPHighRateEnv(**env_kwargs)
    env_states = rollout_env_reference(env, args.steps, seed=0, v_cmd=args.v_cmd)
    ref_states, params_ref = sample_reference_world(
        cmd=args.v_cmd if args.v_cmd is not None else np.mean(env.v_cmd_range),
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        strides=args.strides,
        substeps=args.substeps,
    )

    if args.plot:
        if env_states and ref_states:
            plot_paths(env_states, [np.asarray(s.q, dtype=np.float64) if hasattr(s, "q") else s for s in ref_states], Path(args.plot))
        else:
            print("[plot] missing states; skipping plot")

    if args.visualize:
        ref_world = [np.asarray(s.q, dtype=np.float64) if hasattr(s, "q") else s for s in ref_states]
        _maybe_visualize_dual(env_states, ref_world, env, loop=args.loop)


if __name__ == "__main__":
    main()
