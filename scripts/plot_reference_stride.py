#!/usr/bin/env python
"""
Sample and plot the reference canonical stride (pelvis/swing/stance paths).

Usage:
  python scripts/plot_reference_stride.py --cmd 1.0 --t-ds 0.1 --t-ss 0.6 --strides 6 --substeps 150

Requires: pyvista for plotting. Falls back to printing summary stats if pyvista is unavailable
or a display cannot be opened.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import threelp
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


def sample_reference(cmd: float, t_ds: float, t_ss: float, strides: int, substeps: int):
    params = threelp.ThreeLPParams.Adult()
    res = threelp.build_canonical_stride(cmd, t_ds, t_ss, params)
    if hasattr(res, "success") and not res.success:
        raise RuntimeError("build_canonical_stride failed")
    x = np.asarray(getattr(res, "x_ref"), dtype=np.float64).reshape(-1)
    u_ref = np.asarray(getattr(res, "u_ref"), dtype=np.float64).reshape(-1)
    uv_stride = canonical_action_to_stride(u_ref)

    state_local = lift_canonical_state(x)
    origin = np.zeros(2, dtype=np.float64)
    leg = 1

    pelvis = []
    swing = []
    stance = []

    for _ in range(strides):
        seg = threelp.simulate_stride_with_inputs(
            _to_state(state_local), leg, uv_stride.tolist(), t_ds, t_ss, params, substeps
        )
        for st in seg:
            q_w = _offset_state(np.asarray(st.q, dtype=np.float64), origin)
            pelvis.append(q_w[2:4].copy())
            swing.append(q_w[0:2].copy())
            stance.append(q_w[4:6].copy())
        end_local = np.asarray(seg[-1].q, dtype=np.float64)
        origin = origin + end_local[0:2]  # swing foot becomes new stance origin
        state_local = swap_and_recenter(end_local)
        leg *= -1

    return (
        np.stack(pelvis, axis=0),
        np.stack(swing, axis=0),
        np.stack(stance, axis=0),
    )


def _maybe_plot(pelvis: np.ndarray, swing: np.ndarray, stance: np.ndarray):
    try:
        import pyvista as pv
    except Exception as e:
        print(f"[plot] pyvista unavailable: {e}")
        return
    pts = []
    colors = []
    labels = [("pelvis", pelvis, (0.2, 0.6, 1.0)), ("swing", swing, (0.2, 0.8, 0.2)), ("stance", stance, (0.9, 0.3, 0.2))]
    plotter = pv.Plotter()
    for name, arr, color in labels:
        arr3 = np.hstack([arr, np.zeros((arr.shape[0], 1))])
        line = pv.lines_from_points(arr3, close=False)
        plotter.add_mesh(line, color=color, line_width=3, label=name)
        plotter.add_mesh(pv.Sphere(radius=0.01, center=arr3[0]), color=color)
        plotter.add_mesh(pv.Sphere(radius=0.01, center=arr3[-1]), color=color)
    plotter.add_axes()
    plotter.add_legend()
    try:
        plotter.show()
    except Exception as e:
        print(f"[plot] failed to open window: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot reference canonical stride paths.")
    parser.add_argument("--cmd", type=float, default=1.0, help="Desired speed for build_canonical_stride.")
    parser.add_argument("--t-ds", type=float, default=0.1, help="Double support time.")
    parser.add_argument("--t-ss", type=float, default=0.6, help="Single support time.")
    parser.add_argument("--strides", type=int, default=6, help="Number of strides to sample.")
    parser.add_argument("--substeps", type=int, default=150, help="Dense samples per stride (passed to simulate_stride_with_inputs).")
    args = parser.parse_args()

    try:
        pelvis, swing, stance = sample_reference(args.cmd, args.t_ds, args.t_ss, args.strides, args.substeps)
    except Exception as e:
        print(f"[plot] failed to sample reference: {e}")
        sys.exit(1)

    print(f"[ref] pelvis path shape {pelvis.shape}, swing {swing.shape}, stance {stance.shape}")
    print(f"[ref] pelvis x-range {pelvis[:,0].min():.3f} .. {pelvis[:,0].max():.3f}, y-range {pelvis[:,1].min():.3f} .. {pelvis[:,1].max():.3f}")
    print(f"[ref] swing y-range {swing[:,1].min():.3f} .. {swing[:,1].max():.3f}")

    _maybe_plot(pelvis, swing, stance)


if __name__ == "__main__":
    main()
