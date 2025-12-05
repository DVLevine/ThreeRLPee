#!/usr/bin/env python
"""
Validate the high-rate env's reference pipeline by rolling with zero action
and checking that it stays upright and tracks the stored reference.

Reports:
  - steps before fall/termination
  - max |x_can - x_ref| over the rollout
  - optional XY overlay vs canonical reference (build_canonical_stride replay)

Usage:
  python scripts/validate_env_reference.py --v-cmd 1.0 --steps 400 --plot runs/val_ref.png --visualize
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_high_rate_3lp import ThreeLPHighRateEnv
from scripts.visualize_reference_stride import sample_reference_world  # canonical reference helper


def rollout_env_reference(env: ThreeLPHighRateEnv, steps: int, seed: int | None, v_cmd: float | None, log_every: int = 50):
    options = {}
    if v_cmd is not None:
        options["v_cmd"] = float(v_cmd)
    obs, _ = env.reset(seed=seed, options=options if options else None)
    x_errs = []
    world_states: List[np.ndarray] = []
    fallen = False

    last_log = 0
    for step_idx in range(steps):
        phi = float(env.t_stride / env.stride_time) if env.stride_time > 0 else 0.0
        x_ref = env.reference_state(phi) if hasattr(env, "reference_state") else np.zeros(8, dtype=np.float64)
        x_errs.append(np.abs(env.x_can - x_ref))

        action = np.zeros(env.action_dim, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)
        if "state_world" in info:
            world_states.append(np.asarray(info["state_world"], dtype=np.float64))
        fallen = fallen or bool(info.get("fallen", False))
        if log_every > 0 and (step_idx + 1) % log_every == 0:
            mse = float(np.mean(np.square(x_errs)))
            print(f"[progress] step={step_idx+1} mse(x_can, x_ref)={mse:.4e} fallen={fallen}")
            last_log = step_idx + 1
        if terminated or truncated:
            break

    x_errs = np.asarray(x_errs)
    max_err = float(np.max(x_errs)) if x_errs.size else 0.0
    mse_final = float(np.mean(np.square(x_errs))) if x_errs.size else 0.0
    if log_every > 0 and last_log < len(x_errs):
        print(f"[progress] step={len(x_errs)} mse(x_can, x_ref)={mse_final:.4e} fallen={fallen}")
    return world_states, fallen, max_err, mse_final


def _extract_xy(states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(states)
    pelvis = arr[:, 2:4]
    swing = arr[:, 0:2]
    stance = arr[:, 4:6]
    return pelvis, swing, stance


def plot_overlay(env_states: List[np.ndarray], ref_states: List[np.ndarray], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional plotting
        print(f"[plot] matplotlib unavailable: {e}")
        return
    if not env_states or not ref_states:
        print("[plot] missing states; skipping plot")
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


def maybe_visualize(states: List[np.ndarray], env: ThreeLPHighRateEnv, loop: bool):
    try:
        import threelp  # type: ignore
    except Exception as e:  # pragma: no cover - GUI path
        print(f"[viz] threelp unavailable: {e}")
        return
    if not hasattr(threelp, "visualize_trajectory"):
        print("[viz] visualize_trajectory not available (build with PY_VISUALIZER=ON)")
        return
    if not states:
        print("[viz] no states to visualize")
        return
    try:
        params = env.sim.get_params() if hasattr(env, "sim") else threelp.ThreeLPParams.Adult()
        state_objs = []
        for q in states:
            st = threelp.ThreeLPState()
            st.q = [float(v) for v in q]
            state_objs.append(st)
        threelp.visualize_trajectory(
            state_objs,
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
    p = argparse.ArgumentParser(description="Validate env reference rollout (zero action) vs canonical reference.")
    p.add_argument("--steps", type=int, default=400, help="Max dt steps for env rollout.")
    p.add_argument("--v-cmd", type=float, default=None, help="Optional fixed command speed.")
    p.add_argument("--t-ds", type=float, default=0.1)
    p.add_argument("--t-ss", type=float, default=0.6)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--ref-substeps", type=int, default=None, help="Force reference substeps in env (optional).")
    p.add_argument("--plot", type=str, default=None, help="Optional PNG path for XY overlay.")
    p.add_argument("--visualize", action="store_true", help="Visualize env rollout if available.")
    p.add_argument("--loop", action="store_true", help="Loop visualization playback.")
    p.add_argument("--log-every", type=int, default=50, help="Print running MSE every N steps (0 to disable).")
    args = p.parse_args()

    env_kwargs = dict(t_ds=args.t_ds, t_ss=args.t_ss, dt=args.dt, max_steps=args.max_env_steps)
    if args.ref_substeps is not None:
        env_kwargs["ref_substeps"] = args.ref_substeps
    # Use deterministic phase to remove random warm-start effects.
    env_kwargs["random_phase"] = False
    env = ThreeLPHighRateEnv(**env_kwargs)

    env_states, fallen, max_err, mse_final = rollout_env_reference(
        env, args.steps, seed=0, v_cmd=args.v_cmd, log_every=args.log_every
    )
    cmd_used = args.v_cmd if args.v_cmd is not None else np.mean(env.v_cmd_range)
    ref_states, _ = sample_reference_world(cmd=cmd_used, t_ds=args.t_ds, t_ss=args.t_ss, strides=4, substeps=args.ref_substeps or 150)

    print(f"[validate] steps={len(env_states)} fallen={fallen} max|x_can - x_ref|={max_err:.4e} mse={mse_final:.4e}")

    if args.plot:
        ref_world = [np.asarray(s.q, dtype=np.float64) if hasattr(s, "q") else s for s in ref_states]
        plot_overlay(env_states, ref_world, Path(args.plot))
    if args.visualize:
        maybe_visualize(env_states, env, loop=args.loop)


if __name__ == "__main__":
    main()
