import argparse
import csv
import time
from pathlib import Path

import numpy as np

from three_lp_py import ThreeLPSimPy

try:
    import pyvista as pv
except Exception:  # pragma: no cover - optional dependency
    pv = None


def run_rollout(
    steps: int,
    action_scale: float,
    csv_path: Path,
    visualize: bool,
    seed: int | None,
    t_ds: float,
    t_ss: float,
    dt: float,
    offscreen: bool = False,
):
    rng = np.random.default_rng(seed)
    sim = ThreeLPSimPy(t_ds=t_ds, t_ss=t_ss, dt=dt, max_action=action_scale)
    state = sim.reset()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["t", "phase", "phase_time", "phase_duration", "support_sign"]
            + [f"a{i}" for i in range(sim.action_dim)]
            + [f"x{i}" for i in range(sim.state_dim)]
        )
        writer.writerow(header)

        plotter = None
        handles = {}
        if visualize:
            if pv is None:
                raise RuntimeError("pyvista is not installed; cannot visualize")
            plotter = _init_plotter(offscreen=offscreen)
            handles = _init_scene(plotter)

        t = 0.0
        for _ in range(steps):
            action = rng.normal(scale=action_scale * 0.1, size=(sim.action_dim,))
            state, info = sim.step(action)

            row = [
                t,
                info.get("phase", ""),
                info.get("phase_time", 0.0),
                info.get("phase_duration", 0.0),
                info.get("support_sign", 1.0),
                *action.tolist(),
                *state.tolist(),
            ]
            writer.writerow(row)

            if plotter is not None and not offscreen:
                _update_scene(plotter, handles, state)
                plotter.update()
                time.sleep(sim.dt)

            t += sim.dt

        if plotter is not None:
            plotter.show(interactive=False, auto_close=False)


def _init_plotter(offscreen: bool = False):
    plotter = pv.Plotter(window_size=(800, 600), off_screen=offscreen)
    plotter.add_axes()
    plotter.set_background("white")
    plane = pv.Plane(i_size=1.5, j_size=1.5)
    plotter.add_mesh(plane, color="lightgray", opacity=0.2)
    plotter.show(auto_close=False, interactive_update=True)
    return plotter


def _init_scene(plotter):
    handles = {}
    pts = np.zeros((3, 3))
    poly = pv.PolyData(pts)
    handles["points_poly"] = poly
    handles["points_actor"] = plotter.add_mesh(poly, color="blue", point_size=12, render_points_as_spheres=True)
    # Lines pelvis->feet
    # vtk format: [npts, p0, p1, npts, p2, p3] for two segments (X2-x1, x1-X3)
    lines = np.array([2, 1, 0, 2, 1, 2], dtype=np.int64)
    lines_poly = pv.PolyData(pts, lines=lines)
    handles["lines_poly"] = lines_poly
    handles["lines_actor"] = plotter.add_mesh(lines_poly, color="black", line_width=3)
    return handles


def _update_scene(plotter, handles, state):
    # Extract foot/pelvis horizontal positions, keep z=0 plane for visualization.
    X2 = state[0:2]
    x1 = state[2:4]
    X3 = state[4:6]
    pts = np.vstack([X2, x1, X3])
    pts_3d = np.hstack([pts, np.zeros((3, 1))])

    handles["points_poly"].points = pts_3d
    handles["lines_poly"].points = pts_3d
    handles["points_poly"].modified()
    handles["lines_poly"].modified()


def main():
    parser = argparse.ArgumentParser(description="Roll out ThreeLPSimPy and log CSV (optional PyVista visualization).")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps.")
    parser.add_argument("--dt", type=float, default=0.02, help="Timestep for integration.")
    parser.add_argument("--t-ds", type=float, default=0.1, help="Double support duration.")
    parser.add_argument("--t-ss", type=float, default=0.3, help="Single support duration.")
    parser.add_argument("--action-scale", type=float, default=50.0, help="Action magnitude clip and sampling scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for actions.")
    parser.add_argument("--csv", type=Path, default=Path("rollout_py.csv"), help="Output CSV path.")
    parser.add_argument("--vis", action="store_true", help="Enable PyVista visualization.")
    parser.add_argument("--offscreen", action="store_true", help="Use offscreen rendering (no display).")
    args = parser.parse_args()

    run_rollout(
        steps=args.steps,
        action_scale=args.action_scale,
        csv_path=args.csv,
        visualize=args.vis,
        seed=args.seed if args.seed >= 0 else None,
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        dt=args.dt,
        offscreen=args.offscreen,
    )


if __name__ == "__main__":
    main()
