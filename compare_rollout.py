"""
Replay a saved action CSV through the Python 3LP sim (and an optional
pybind-backed C++ sim if available) to compare trajectories.

CSV format expected: produced by simulate_3lp_py.py
  t, phase, phase_time, phase_duration, support_sign, a0..a7, x0..x11
Only the action columns are consumed.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from three_lp_py import ThreeLPSimPy


def load_actions(csv_path: Path, action_dim: int) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = np.array([float(row[f"a{i}"]) for i in range(action_dim)], dtype=np.float64)
            actions.append(a)
    return actions


def run_sim(sim, actions: List[np.ndarray]) -> np.ndarray:
    state = sim.reset()
    states = [state.copy()]
    infos = []
    for a in actions:
        state, info = sim.step(a)
        states.append(state.copy())
        infos.append(info)
    return np.stack(states, axis=0), infos


def try_import_cpp_sim():
    try:
        from three_lp_cpp import ThreeLPSim  # type: ignore
    except Exception:
        return None
    return ThreeLPSim


def main():
    parser = argparse.ArgumentParser(description="Compare Python 3LP sim vs optional C++ pybind sim on a saved action sequence.")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV with actions (from simulate_3lp_py.py).")
    parser.add_argument("--out", type=Path, default=Path("compare_out.csv"), help="Output CSV with states and deltas.")
    parser.add_argument("--dt", type=float, default=0.02, help="Sim timestep.")
    parser.add_argument("--t-ds", type=float, default=0.1, help="Double support duration.")
    parser.add_argument("--t-ss", type=float, default=0.3, help="Single support duration.")
    args = parser.parse_args()

    # Python sim
    py_sim = ThreeLPSimPy(dt=args.dt, t_ds=args.t_ds, t_ss=args.t_ss)
    actions = load_actions(args.csv, action_dim=py_sim.action_dim)

    py_states, _ = run_sim(py_sim, actions)

    cpp_cls = try_import_cpp_sim()
    cpp_states = None
    if cpp_cls is not None:
        cpp_sim = cpp_cls(dt=args.dt, t_ds=args.t_ds, t_ss=args.t_ss)
        cpp_states, _ = run_sim(cpp_sim, actions)

    # Write comparison
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["step", "t"] + [f"py_x{i}" for i in range(py_states.shape[1])]
        if cpp_states is not None:
            header += [f"cpp_x{i}" for i in range(cpp_states.shape[1])]
            header += [f"diff_x{i}" for i in range(py_states.shape[1])]
        writer.writerow(header)

        for k in range(py_states.shape[0]):
            t = k * args.dt
            row = [k, t] + py_states[k].tolist()
            if cpp_states is not None:
                row += cpp_states[k].tolist()
                row += (py_states[k] - cpp_states[k]).tolist()
            writer.writerow(row)

    # Print RMS summary
    if cpp_states is not None:
        diffs = py_states[: cpp_states.shape[0]] - cpp_states
        rms = np.sqrt(np.mean(diffs ** 2, axis=0))
        print("Comparison vs C++ pybind sim:")
        print(f"  steps compared: {cpp_states.shape[0]}")
        print(f"  per-state-dim RMS error: {rms}")
        print(f"  overall RMS norm: {np.sqrt(np.mean(diffs ** 2)):.6f}")
    else:
        print("C++ pybind sim not available (module three_lp_cpp not found). Wrote Python states only.")
        print(f"Actions replayed: {len(actions)}; states saved to {args.out}")


if __name__ == "__main__":
    main()
