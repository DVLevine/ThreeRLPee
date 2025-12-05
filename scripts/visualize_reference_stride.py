#!/usr/bin/env python
"""
Visualize the open-loop reference gait from build_canonical_stride in the Open3D viewer.

Usage:
  python scripts/visualize_reference_stride.py --cmd 1.0 --t-ds 0.1 --t-ss 0.6 --strides 4 --substeps 150 --loop

If visualize_trajectory is unavailable (pybind built without visualizer or no display),
prints a short summary instead.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import threelp  # type: ignore
except Exception as e:  # pragma: no cover - manual tool
    print(f"[viz] threelp import failed: {e}")
    sys.exit(1)

from stride_utils import (
    canonical_action_to_stride,
    lift_canonical_state,
    swap_and_recenter,
    _offset_state,  # type: ignore
)


def _to_state(q: np.ndarray):
    st = threelp.ThreeLPState()
    st.q = [float(v) for v in q.reshape(-1)]
    return st


def sample_reference_world(cmd: float, t_ds: float, t_ss: float, strides: int, substeps: int):
    params = threelp.ThreeLPParams.Adult()
    ref = threelp.build_canonical_stride(cmd, t_ds, t_ss, params)
    if hasattr(ref, "success") and not ref.success:
        raise RuntimeError("build_canonical_stride failed")
    x = np.asarray(getattr(ref, "x_ref"), dtype=np.float64).reshape(-1)
    u_ref = np.asarray(getattr(ref, "u_ref"), dtype=np.float64).reshape(-1)
    uv_stride = canonical_action_to_stride(u_ref)

    state_local = lift_canonical_state(x)
    origin = np.zeros(2, dtype=np.float64)
    leg = 1

    world_states = []

    for _ in range(strides):
        seg = threelp.simulate_stride_with_inputs(
            _to_state(state_local), leg, uv_stride.tolist(), t_ds, t_ss, params, substeps
        )
        for st in seg:
            q_w = _offset_state(np.asarray(st.q, dtype=np.float64), origin)
            world_states.append(_to_state(q_w))
        end_local = np.asarray(seg[-1].q, dtype=np.float64)
        origin = origin + end_local[0:2]  # swing foot becomes new stance origin
        state_local = swap_and_recenter(end_local)
        leg *= -1

    return world_states, params


def main():
    parser = argparse.ArgumentParser(description="Visualize reference gait (open-loop) using visualize_trajectory.")
    parser.add_argument("--cmd", type=float, default=1.0, help="Desired speed for build_canonical_stride.")
    parser.add_argument("--t-ds", type=float, default=0.1, help="Double support time.")
    parser.add_argument("--t-ss", type=float, default=0.6, help="Single support time.")
    parser.add_argument("--strides", type=int, default=4, help="Number of strides to sample.")
    parser.add_argument("--substeps", type=int, default=150, help="Dense samples per stride.")
    parser.add_argument("--loop", action="store_true", help="Loop playback in the viewer.")
    args = parser.parse_args()

    if not hasattr(threelp, "visualize_trajectory"):
        print("[viz] visualize_trajectory not available (build with PY_VISUALIZER=ON)")
        sys.exit(0)

    try:
        states, params = sample_reference_world(args.cmd, args.t_ds, args.t_ss, args.strides, args.substeps)
    except Exception as e:
        print(f"[viz] failed to build reference: {e}")
        sys.exit(1)

    print(f"[viz] sampled {len(states)} states over {args.strides} strides at cmd={args.cmd}")
    try:
        threelp.visualize_trajectory(
            states,
            args.t_ds,
            args.t_ss,
            params,
            fps=60.0,
            loop=args.loop,
            wait_for_close=args.loop,
        )
    except Exception as e:  # pragma: no cover - GUI path
        print(f"[viz] visualize_trajectory error: {e}")


if __name__ == "__main__":
    main()
