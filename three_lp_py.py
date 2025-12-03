"""
Deprecated Python 3LP implementation.

Use the C++ pybind module `threelp` instead. This file is kept in the repo for
reference only and will raise at import time to avoid accidental usage.
"""
raise RuntimeError("three_lp_py.py is deprecated; import threelp (C++ pybind) instead.")

import math  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from typing import Dict, Tuple  # noqa: F401

import numpy as np  # noqa: F401

try:  # noqa: F401
    from scipy.linalg import expm as _scipy_expm  # type: ignore # noqa: F401
except Exception:  # pragma: no cover - scipy may not be installed
    _scipy_expm = None  # noqa: F401


def _matrix_exponential(mat: np.ndarray) -> np.ndarray:
    """
    Compute exp(mat) with scipy if available. If scipy is missing, fall back to
    an eigen-decomposition with a Moore-Penrose pseudo-inverse to tolerate
    near-singular eigenvector matrices.
    """
    if _scipy_expm is not None:
        return _scipy_expm(mat)
    # Fallback: V exp(Λ) V^{-1} (or V^{+} if singular)
    w, v = np.linalg.eig(mat)
    try:
        vinv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        vinv = np.linalg.pinv(v)
    exp_w = np.exp(w)
    return v @ np.diag(exp_w) @ vinv


@dataclass
class ThreeLPParams:
    h1: float  # pelvis plane height
    h2: float  # leg mass plane height
    h3: float  # torso mass plane height above pelvis plane
    wP: float  # pelvis width
    m1: float  # torso mass
    m2: float  # each leg mass
    g: float   # gravity

    @staticmethod
    def Adult() -> "ThreeLPParams":
        # Matches ThreeLPee/include/three_lp_params.hpp defaults
        return ThreeLPParams(
            h1=0.89,
            h2=0.32,
            h3=0.36,
            wP=0.20,
            m1=45.7,
            m2=12.15,
            g=9.81,
        )


@dataclass
class PhaseMatrices:
    C_X: np.ndarray
    C_U: np.ndarray
    A_V0: np.ndarray
    A_V1: np.ndarray
    C_W: np.ndarray
    C_d: np.ndarray
    F0: np.ndarray
    F1: np.ndarray


@dataclass
class PhaseDiscrete:
    A: np.ndarray
    B: np.ndarray


def make3lp_ss(P: ThreeLPParams, t_ss: float, d_sign: float = 1.0) -> PhaseMatrices:
    """
    Build single-support continuous matrices (Appendix A). Time-affine torque
    ramps are encoded via A_V1 = (1/t_ss) * C_U.
    """
    u = P.h3 + P.h1
    v = P.h2 - P.h1
    c1 = P.m1 * u * P.h1 + P.m2 * v * v
    c2 = P.m1 * u * P.h1 + 2.0 * P.m2 * v * v
    c3 = -P.m1 * P.h1 + P.m2 * v - P.m2 * P.h1

    I2 = np.eye(2)

    CX = np.zeros((6, 6))
    coeff = P.g / c1
    CX[0:2, 0:2] = coeff * ((-c2 / P.h2 + P.m2 * v) * I2)
    CX[0:2, 2:4] = coeff * ((P.m1 * P.h1 * (v + u) / P.h2) * I2)
    CX[0:2, 4:6] = coeff * ((v * c3 / P.h2) * I2)
    CX[2:4, 0:2] = coeff * ((P.m2 * P.h1) * I2)
    CX[2:4, 2:4] = coeff * ((P.m1 * P.h1 - P.m2 * v) * I2)
    CX[2:4, 4:6] = coeff * (c3 * I2)

    CU = np.zeros((6, 4))
    CU[0:2, 0:2] = (P.h1 / c1) * ((c2 / P.h2) * I2)
    CU[0:2, 2:4] = (P.h1 / c1) * ((2.0 * P.m2) * I2)
    CU[2:4, 0:2] = (P.h1 / c1) * ((v / P.h2) * I2)
    CU[2:4, 2:4] = (P.h1 / c1) * ((-1.0) * I2)

    CW = np.zeros((6, 4))
    CW[0:2, 0:2] = (P.h1 / c1) * ((v * u / P.h2) * I2)
    CW[0:2, 2:4] = (P.h1 / c1) * ((-v / P.h2) * I2)
    CW[2:4, 0:2] = (P.h1 / c1) * (u * I2)
    CW[2:4, 2:4] = (P.h1 / c1) * ((-1.0) * I2)

    Cd = np.zeros((6, 1))
    J = np.array([[0.0], [1.0]])
    Cd[0:2, 0:1] = (P.g * P.wP / c1) * ((c2 / P.h2) * J)
    Cd[2:4, 0:1] = (P.g * P.wP / c1) * ((P.m2 * v) * J)

    return PhaseMatrices(
        C_X=CX,
        C_U=CU,
        A_V0=np.zeros((6, 4)),
        A_V1=(1.0 / t_ss) * CU,
        C_W=CW,
        C_d=Cd * d_sign,
        F0=np.zeros((6, 6)),
        F1=np.zeros((6, 6)),
    )


def make3lp_ds(P: ThreeLPParams, t_ds: float) -> PhaseMatrices:
    """
    Build double-support continuous matrices. Torques are ramped down from
    the previous stance to the next (A_V0 = C_U, A_V1 = -C_U / t_ds).
    """
    u = P.h3 + P.h1
    v = P.h2 - P.h1
    c2 = P.m1 * u * P.h1 + 2.0 * P.m2 * v * v
    I2 = np.eye(2)

    CU = np.zeros((6, 4))
    CU[2:4, 2:4] = (P.h1 / c2) * I2

    CW = np.zeros((6, 4))
    CW[2:4, 0:2] = (P.h1 / c2) * (u * I2)
    CW[2:4, 2:4] = (P.h1 / c2) * ((-1.0) * I2)

    k1 = P.g * P.h1 * P.m1 / c2
    k2 = P.g * P.m2 / c2

    CX = np.zeros((6, 6))
    CX[2:4, 2:4] = (k1 - 2.0 * k2 * v) * I2

    F0 = np.zeros((6, 6))
    F1 = np.zeros((6, 6))
    b20 = (-k1 + k2 * (-P.h1 + v))
    b21 = (k1 / t_ds) + (2.0 * k2 * P.h1 / t_ds)
    b30 = (k2 * (-P.h1 + v))
    b31 = (-k1 / t_ds) + (2.0 * k2 * P.h1 / t_ds)
    F0[2:4, 0:2] = b20 * I2
    F0[2:4, 4:6] = b30 * I2
    F1[2:4, 0:2] = b21 * I2
    F1[2:4, 4:6] = b31 * I2

    return PhaseMatrices(
        C_X=CX,
        C_U=CU,
        A_V0=CU,
        A_V1=(-1.0 / t_ds) * CU,
        C_W=CW,
        C_d=np.zeros((6, 1)),
        F0=F0,
        F1=F1,
    )


def compute_exp_phi(A: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicates the augmented exp block used in the C++ code."""
    n = A.shape[0]
    block = np.zeros((3 * n, 3 * n))
    block[0:n, 0:n] = A
    block[0:n, n:2 * n] = np.eye(n)
    block[n:2 * n, 2 * n : 3 * n] = np.eye(n)
    exp_block = _matrix_exponential(block * t)
    expA = exp_block[0:n, 0:n]
    phi1 = exp_block[0:n, n:2 * n]
    phi2 = exp_block[0:n, 2 * n : 3 * n]
    return expA, phi1, phi2


def discretize_phase(PM: PhaseMatrices, t_phase: float) -> PhaseDiscrete:
    """
    Exact discrete map for a given phase duration using the augmented
    exponential trick from StrideBuilder.
    """
    nx = 6
    nQ = 2 * nx
    if t_phase <= 0.0:
        return PhaseDiscrete(A=np.eye(nQ), B=np.zeros((nQ, 13)))

    A = np.zeros((nQ, nQ))
    A[0:nx, nx:2 * nx] = np.eye(nx)
    A[nx:2 * nx, 0:nx] = PM.C_X

    expA, phi1, phi2 = compute_exp_phi(A, t_phase)

    def lift(src: np.ndarray) -> np.ndarray:
        out = np.zeros((nQ, src.shape[1]))
        out[nx:2 * nx, :] = src
        return out

    BU0 = lift(PM.C_U)
    BV0 = lift(PM.A_V0)
    BV1 = lift(PM.A_V1)
    BW0 = lift(PM.C_W)
    Bd0 = lift(PM.C_d)

    B = np.zeros((nQ, 13))
    B[:, 0:4] = phi1 @ BU0
    B[:, 4:8] = phi1 @ BV0 + phi2 @ BV1
    B[:, 8:12] = phi1 @ BW0
    B[:, 12:13] = phi1 @ Bd0

    return PhaseDiscrete(A=expA, B=B)


def enforce_ds_constraints(D: PhaseDiscrete) -> None:
    """
    Clamp foot positions and velocities during double support, matching the C++
    helper.
    """
    def keep_position(row: int) -> None:
        D.A[row, :] = 0.0
        D.A[row, row] = 1.0
        D.B[row, :] = 0.0

    def zero_velocity(row: int) -> None:
        D.A[row, :] = 0.0
        D.B[row, :] = 0.0

    keep_position(0)
    keep_position(1)
    keep_position(4)
    keep_position(5)
    zero_velocity(6)
    zero_velocity(7)
    zero_velocity(10)
    zero_velocity(11)


def enforce_ss_constraints(D: PhaseDiscrete) -> None:
    """
    Keep the stance foot fixed during single support.
    """
    def keep_position(row: int) -> None:
        D.A[row, :] = 0.0
        D.A[row, row] = 1.0
        D.B[row, :] = 0.0

    def zero_velocity(row: int) -> None:
        D.A[row, :] = 0.0
        D.B[row, :] = 0.0

    keep_position(4)
    keep_position(5)
    zero_velocity(10)
    zero_velocity(11)


class ThreeLPSimPy:
    """
    Minimal stateful 3LP simulator in Python.

    - State Q: [X2x, X2y, x1x, x1y, X3x, X3y, X2dx, X2dy, x1dx, x1dy, X3dx, X3dy]
    - Action drives the 4 hip/ankle inputs (U) and the 4 ramped torques (V).
      W is zero; d is set from the stance sign and flips every stride.
    """

    def __init__(
        self,
        params: ThreeLPParams | None = None,
        dt: float = 0.02,
        t_ds: float = 0.1,
        t_ss: float = 0.3,
        max_action: float = 50.0,
    ) -> None:
        self.params = params or ThreeLPParams.Adult()
        self.dt = float(dt)
        self.t_ds = float(t_ds)
        self.t_ss = float(t_ss)
        self.max_action = float(max_action)

        self.state_dim = 12
        self.action_dim = 8  # U (4) + V (4)
        self.phase_durations = {"ds": self.t_ds, "ss": self.t_ss}

        # Precompute discrete maps for a single control step.
        self._D_ds = discretize_phase(make3lp_ds(self.params, self.t_ds), self.dt)
        self._D_ss = discretize_phase(make3lp_ss(self.params, self.t_ss), self.dt)
        enforce_ds_constraints(self._D_ds)
        enforce_ss_constraints(self._D_ss)

        self.reset()

    def reset(
        self,
        state0: np.ndarray | None = None,
        support_sign: float = 1.0,
    ) -> np.ndarray:
        self.state = np.zeros(self.state_dim, dtype=np.float64)
        if state0 is not None:
            state0 = np.asarray(state0, dtype=np.float64)
            if state0.shape[0] != self.state_dim:
                raise ValueError(f"state0 must have shape ({self.state_dim},)")
            self.state = state0.copy()

        self.phase = "ds"
        self.phase_time = 0.0
        self.support_sign = float(math.copysign(1.0, support_sign))
        self.stride_time = self.t_ds + self.t_ss
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] != self.action_dim:
            raise ValueError(f"action must have shape ({self.action_dim},)")
        act = np.clip(act, -self.max_action, self.max_action)

        r = np.zeros(13, dtype=np.float64)
        r[0:4] = act[0:4]          # U
        r[4:8] = act[4:8] if act.shape[0] > 4 else 0.0  # V
        r[12] = self.support_sign

        D = self._D_ds if self.phase == "ds" else self._D_ss
        self.state = D.A @ self.state + D.B @ r

        self.phase_time += self.dt
        current_duration = self.phase_durations[self.phase]
        while self.phase_time >= current_duration and current_duration > 0.0:
            self.phase_time -= current_duration
            if self.phase == "ds":
                self.phase = "ss"
                current_duration = self.phase_durations[self.phase]
            else:
                self.phase = "ds"
                self.support_sign *= -1.0
                current_duration = self.phase_durations[self.phase]

        info = {
            "phase": self.phase,
            "phase_time": self.phase_time,
            "phase_duration": current_duration,
            "support_sign": self.support_sign,
            "fallen": False,
            "dt": self.dt,
        }
        return self.state.copy(), info
