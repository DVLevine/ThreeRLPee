#!/usr/bin/env python
"""
Canonical input sanity check and optional visualization.

For each stride-order input channel, apply a positive and negative pulse from
the reference pose and report the change in canonical reduced coordinates.
Optionally visualize one channel/stance as a trajectory.

Stride-order inputs: [Uh_y, Uh_x, Ua_y, Ua_x, Vh_y, Vh_x, Va_y, Va_x]
"""

import argparse
from typing import List

import numpy as np

try:
    import threelp  # type: ignore
except Exception as exc:  # pragma: no cover - diagnostic script
    raise SystemExit(f"[sanity] threelp import failed: {exc}")

INPUT_NAMES = ["Uh_y", "Uh_x", "Ua_y", "Ua_x", "Vh_y", "Vh_x", "Va_y", "Va_x"]


def to_state(q: np.ndarray):
    st = threelp.ThreeLPState()
    st.q = [float(v) for v in q]
    return st


def mirror_lateral_components(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    for idx in (1, 3, 5, 7, 9, 11):
        q[idx] = -q[idx]
    return q


def canonicalize_reduced(st, support_sign: int) -> np.ndarray:
    return np.asarray(threelp.canonicalize_reduced_state(st.q, support_sign), dtype=np.float64)


def run_pulse(leg_flag: int, idx: int, amp: float, q0_ref: np.ndarray, params, t_ds: float, t_ss: float) -> np.ndarray:
    sim = threelp.ThreeLPSim(t_ds, t_ss, params, True)
    q0 = q0_ref if leg_flag > 0 else mirror_lateral_components(q0_ref)
    sim.reset(to_state(q0), leg_flag)
    action = np.zeros(8, dtype=np.float64)
    action[idx] = amp
    st1, info1 = sim.step_closed_form(action.tolist())
    return canonicalize_reduced(st1, info1["support_sign"])


def build_trajectory(
    leg_flag: int,
    idx: int,
    amp: float,
    q0_ref: np.ndarray,
    params,
    t_ds: float,
    t_ss: float,
    substeps: int,
) -> List[object]:
    sim = threelp.ThreeLPSim(t_ds, t_ss, params, True)
    q0 = q0_ref if leg_flag > 0 else mirror_lateral_components(q0_ref)
    sim.reset(to_state(q0), leg_flag)
    action = np.zeros(8, dtype=np.float64)
    action[idx] = amp
    states = [sim.get_state_world()]
    stride_time = t_ds + t_ss
    dt = stride_time / max(1, substeps)
    t = 0.0
    while t < stride_time - 1e-12:
        sim.step_dt(action.tolist(), dt)
        states.append(sim.get_state_world())
        t += dt
    return states


def visualize(states: List[object], t_ds: float, t_ss: float, params, fps: float, loop: bool) -> None:
    if not hasattr(threelp, "visualize_trajectory"):
        print("[sanity] visualize_trajectory not available in threelp build; skipping viz")
        return
    try:
        threelp.visualize_trajectory(states, t_ds=t_ds, t_ss=t_ss, params=params, fps=fps, loop=loop, wait_for_close=True)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"[sanity] visualization error: {exc}")


def main():
    ap = argparse.ArgumentParser(description="Canonical input sanity pulses and optional visualization.")
    ap.add_argument("--v-cmd", type=float, default=1.0, help="Reference command speed for q0.")
    ap.add_argument("--t-ds", type=float, default=0.1)
    ap.add_argument("--t-ss", type=float, default=0.6)
    ap.add_argument("--amp", type=float, default=20.0, help="Pulse amplitude.")
    ap.add_argument("--substeps", type=int, default=120, help="Substeps for visualization trajectories.")
    ap.add_argument("--visualize", action="store_true", help="Visualize a single pulse trajectory.")
    ap.add_argument("--input", type=str, default=None, help="Input name to visualize (e.g., Ua_y).")
    ap.add_argument("--stance", type=int, default=1, choices=[1, -1], help="Stance to visualize (+1 left, -1 right).")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    params = threelp.ThreeLPParams.Adult()
    ref = threelp.sample_reference_stride(args.v_cmd, args.t_ds, args.t_ss, params, max(10, args.substeps))
    q_ref0 = np.asarray(ref["q_ref0"], dtype=np.float64)

    print(f"[sanity] stride-order pulse amplitude={args.amp}")
    print(f"[sanity] inputs: {INPUT_NAMES}")
    print(f"[sanity] starting from reference q0 (v_cmd={args.v_cmd}), stance recentered")

    for leg_flag in (+1, -1):
        print(f"\n[stance {leg_flag:+d}]")
        for idx, name in enumerate(INPUT_NAMES):
            x_pos = run_pulse(leg_flag, idx, +args.amp, q_ref0, params, args.t_ds, args.t_ss)
            x_neg = run_pulse(leg_flag, idx, -args.amp, q_ref0, params, args.t_ds, args.t_ss)
            delta = x_pos - x_neg
            print(
                f"{name:5s}  Δs1=({delta[0]: .3f},{delta[1]: .3f}) "
                f"Δs2=({delta[2]: .3f},{delta[3]: .3f}) "
                f"Δds1=({delta[4]: .3f},{delta[5]: .3f}) "
                f"Δds2=({delta[6]: .3f},{delta[7]: .3f})"
            )

    if args.visualize:
        if args.input is None:
            print("[sanity] --input is required when --visualize is set (one of: %s)" % ", ".join(INPUT_NAMES))
            return
        try:
            idx = INPUT_NAMES.index(args.input)
        except ValueError:
            raise SystemExit(f"[sanity] invalid --input {args.input}; choose from {INPUT_NAMES}")
        states = build_trajectory(args.stance, idx, args.amp, q_ref0, params, args.t_ds, args.t_ss, args.substeps)
        print(f"[sanity] visualizing input {args.input} stance {args.stance:+d} amp={args.amp}")
        visualize(states, args.t_ds, args.t_ss, params, args.fps, args.loop)


if __name__ == "__main__":  # pragma: no cover - diagnostic script
    main()
