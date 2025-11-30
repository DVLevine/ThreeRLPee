#!/usr/bin/env python
"""
Simple demo to exercise threelp.visualize_trajectory.
Generates a synthetic walking-like trajectory and opens the Open3D window.
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import List

try:
    import threelp
except Exception as exc:  # pragma: no cover
    print(f"Failed to import threelp: {exc}", file=sys.stderr)
    sys.exit(1)


def make_states(steps: int) -> List["threelp.ThreeLPState"]:
    """
    Build a simple forward stride sequence.
    q layout (positions only): swing (x,y), pelvis (x,y), stance (x,y), then velocities.
    """
    params = threelp.ThreeLPParams.Adult()
    states: List[threelp.ThreeLPState] = []
    for i in range(steps):
        u = i / max(1, steps - 1)
        swing_x = 0.2 + 0.3 * u
        swing_y = 0.05 * math.sin(2 * math.pi * u)
        pelvis_x = 0.1 + 0.25 * u
        pelvis_y = 0.02 * math.sin(4 * math.pi * u)
        stance_x = -0.1 + 0.05 * u
        stance_y = -0.02 * math.sin(4 * math.pi * u)

        st = threelp.ThreeLPState()
        st.q = [
            swing_x,
            swing_y,
            pelvis_x,
            pelvis_y,
            stance_x,
            stance_y,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        states.append(st)
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo Open3D visualizer for threelp")
    parser.add_argument("--steps", type=int, default=120, help="frames to display")
    parser.add_argument("--fps", type=float, default=60.0, help="frames per second")
    args = parser.parse_args()

    params = threelp.ThreeLPParams.Adult()

    if not hasattr(threelp, "visualize_trajectory"):
        print(
            "visualize_trajectory not available; rebuild with PY_VISUALIZER=ON",
            file=sys.stderr,
        )
        return 1

    states = make_states(args.steps)
    threelp.visualize_trajectory(states, t_ds=0.1, t_ss=0.3, params=params, fps=args.fps)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
