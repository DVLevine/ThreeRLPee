import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import threelp  # type: ignore
except Exception:
    threelp = None

# Map canonical stride actions [Uh_y, Ua_y, Vh_y, Va_y, Uh_x, Ua_x, Vh_x, Va_x]
# back to the stride-map order expected by ThreeLPSim/pybind utilities:
# [Uh_y, Uh_x, Ua_y, Ua_x, Vh_y, Vh_x, Va_y, Va_x].
_CANONICAL_TO_STRIDE = (0, 2, 4, 6, 1, 3, 5, 7)


def canonical_action_to_stride(action: Sequence[float]) -> np.ndarray:
    stride = np.zeros(8, dtype=np.float64)
    for i, col in enumerate(_CANONICAL_TO_STRIDE):
        if i < len(action) and col < len(stride):
            stride[col] = float(action[i])
    return stride


def lift_canonical_state(x: Sequence[float]) -> np.ndarray:
    """Embed 8-D canonical state back into a 12-D stance frame (stance at origin)."""
    q = np.zeros(12, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 8:
        raise ValueError("canonical state must have length 8")
    q[2], q[3] = x[0], x[1]  # pelvis position
    q[8], q[9] = x[2], x[3]  # pelvis velocity
    q[0], q[1] = x[4], x[5]  # swing position
    q[6], q[7] = x[6], x[7]  # swing velocity
    # stance position (4,5) stays at zero; stance velocity (10,11) zero.
    return q


def _swap_matrix() -> np.ndarray:
    S = np.zeros((12, 12), dtype=np.float64)
    I = np.eye(2, dtype=np.float64)
    S[0:2, 4:6] = I
    S[2:4, 2:4] = I
    S[4:6, 0:2] = I
    S[6:8, 10:12] = I
    S[8:10, 8:10] = I
    S[10:12, 6:8] = I
    return S


def _recenter_matrix() -> np.ndarray:
    C = np.eye(12, dtype=np.float64)
    C[0, 4] -= 1.0
    C[1, 5] -= 1.0
    C[2, 4] -= 1.0
    C[3, 5] -= 1.0
    C[4, :] = 0.0
    C[5, :] = 0.0
    return C


_S_Q = _swap_matrix()
_C_RECENTER = _recenter_matrix()


def swap_and_recenter(q: np.ndarray) -> np.ndarray:
    """Apply the C * S transform used by ThreeLPSim between stances."""
    return _C_RECENTER @ (_S_Q @ q.reshape(12, 1)).reshape(-1)


def _offset_state(state_q: np.ndarray, origin_xy: np.ndarray) -> np.ndarray:
    """Translate a stance-frame state by the world origin for visualization."""
    out = np.array(state_q, dtype=np.float64)
    if origin_xy.shape != (2,):
        origin_xy = np.asarray(origin_xy, dtype=np.float64).reshape(-1)[:2]
    offset = np.tile(origin_xy, 3)
    out[:6] += offset
    return out


def _to_state_obj(q: np.ndarray):
    s = threelp.ThreeLPState()
    s.q = [float(v) for v in q]
    return s


@dataclass
class RolloutResult:
    states: List  # list of threelp.ThreeLPState with world-frame positions
    t_ds: float
    t_ss: float
    params: object


def rollout_canonical_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env_kwargs: dict,
    max_steps: int = 50,
    n_substeps: int = 120,
    seed: Optional[int] = None,
    log_prefix: str = "viz",
) -> Optional[RolloutResult]:
    """
    Roll out a policy in the canonical stride env and collect world-frame states
    suitable for threelp.visualize_trajectory. Does not block on rendering.
    """
    if threelp is None:
        print(f"[{log_prefix}] skip: threelp module not available")
        return None
    try:
        from env_canonical_stride import ThreeLPCanonicalStrideEnv
    except Exception as e:
        print(f"[{log_prefix}] skip: failed to import env ({e})")
        return None

    env = ThreeLPCanonicalStrideEnv(**env_kwargs, seed=seed)
    obs, _ = env.reset()
    # Start in the stance frame.
    x_abs = env.current.x_ref + env.delta_x
    state_local = lift_canonical_state(x_abs)
    leg_flag = 1
    origin = np.zeros(2, dtype=np.float64)

    world_states: List = []

    for _ in range(max_steps):
        action = np.asarray(policy_fn(obs), dtype=np.float64).reshape(-1)
        total_action = env.current.u_ref + action
        stride_action = canonical_action_to_stride(total_action)

        try:
            seg = threelp.simulate_stride_with_inputs(
                _to_state_obj(state_local),
                leg_flag,
                stride_action.tolist(),
                env.t_ds,
                env.t_ss,
                env.params,
                n_substeps,
            )
        except Exception as e:
            print(f"[{log_prefix}] simulate_stride_with_inputs error: {e}")
            break

        # Offset to world frame for visualization.
        for st in seg:
            q = _offset_state(np.asarray(st.q, dtype=np.float64), origin)
            world_states.append(_to_state_obj(q))

        end_local = np.asarray(seg[-1].q, dtype=np.float64)
        origin = origin + end_local[0:2]  # swing foot becomes new stance origin
        state_local = swap_and_recenter(end_local)
        leg_flag *= -1

        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            break

    return RolloutResult(states=world_states, t_ds=env.t_ds, t_ss=env.t_ss, params=env.params)


def visualize_rollout(
    rollout: RolloutResult,
    loop: bool = False,
    fps: float = 60.0,
    log_prefix: str = "viz",
    goal: Optional[Sequence[float]] = None,
) -> None:
    if threelp is None or not hasattr(threelp, "visualize_trajectory"):
        print(f"[{log_prefix}] skip: visualize_trajectory not available")
        return
    if rollout is None or not rollout.states:
        print(f"[{log_prefix}] skip: empty rollout")
        return
    kwargs = {}
    if goal is not None:
        if len(goal) == 2:
            kwargs["goal"] = (float(goal[0]), float(goal[1]), 0.0)
        elif len(goal) >= 3:
            kwargs["goal"] = tuple(float(v) for v in goal[:3])
    try:
        threelp.visualize_trajectory(
            rollout.states,
            rollout.t_ds,
            rollout.t_ss,
            rollout.params,
            fps=fps,
            loop=loop,
            wait_for_close=loop,
            **kwargs,
        )
    except Exception as e:
        print(f"[{log_prefix}] visualize_trajectory error: {e}")


def render_canonical_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env_kwargs: dict,
    max_steps: int = 50,
    n_substeps: int = 120,
    seed: Optional[int] = None,
    loop: bool = False,
    log_prefix: str = "viz",
    goal: Optional[Sequence[float]] = None,
) -> None:
    rollout = rollout_canonical_policy(
        policy_fn,
        env_kwargs,
        max_steps=max_steps,
        n_substeps=n_substeps,
        seed=seed,
        log_prefix=log_prefix,
    )
    visualize_rollout(rollout, loop=loop, log_prefix=log_prefix, goal=goal)


def make_run_dir(prefix: str = "stride", base: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base) / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict):
        rec = dict(record)
        rec.setdefault("timestamp", datetime.now().isoformat())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
