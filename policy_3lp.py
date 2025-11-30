# policy_3lp.py
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class PolicyConfig:
    obs_dim: int
    action_dim: int
    basis_dim: int = 64  # legacy/default for MLP paths
    actor_basis_dim: int = 15
    critic_basis_dim: int = 22
    encoder_type: str = "raw"  # "raw", "goal_walk", "vel_walk"
    log_std_init: float = -0.5
    hidden_sizes: tuple[int, ...] = (64, 64)


class BasisEncoder(nn.Module):
    """
    3LP-inspired structured basis.

    Actor features (15 dims):
      [1, s1_norm (2), s2_norm (2), v1_norm (2), v2_norm (2), g_rel_norm (2),
       sigma_bar, sigma_bar^2, sin(pi*sigma), cos(pi*sigma)]

    Critic features (22 dims):
      [1, z (10 linear terms), q1..q11 quadratic terms]
    """

    def __init__(self, obs_dim: int, actor_basis_dim: int, critic_basis_dim: int, state_dim_3lp: int, goal_dim: int = 2, leg_length: float = 0.89, gravity: float = 9.81):
        super().__init__()
        self.obs_dim = obs_dim
        self.actor_basis_dim = actor_basis_dim
        self.critic_basis_dim = critic_basis_dim
        self.state_dim_3lp = state_dim_3lp
        self.goal_dim = goal_dim
        self.leg_length = leg_length
        self.omega0 = (gravity / leg_length) ** 0.5

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, obs_dim] with layout [state(12), goal_rel_pelvis(2), phase_frac(1)]
        returns φ_actor: [B, actor_basis_dim]
        """
        state = obs[..., : self.state_dim_3lp]
        goal_rel_pelvis = obs[..., self.state_dim_3lp : self.state_dim_3lp + self.goal_dim]
        phase = obs[..., -1:]  # normalized phase fraction in [0,1]

        X2 = state[..., 0:2]
        x1 = state[..., 2:4]
        X3 = state[..., 4:6]
        X2dot = state[..., 6:8]
        x1dot = state[..., 8:10]

        s1 = x1 - X3
        s2 = X2 - X3
        v1 = x1dot
        v2 = X2dot
        g_rel = goal_rel_pelvis + s1  # goal relative to stance foot

        L = self.leg_length
        w0 = self.omega0

        s1n = s1 / L
        s2n = s2 / L
        g_n = g_rel / L
        v1n = v1 / (L * w0)
        v2n = v2 / (L * w0)

        sigma = torch.clamp(phase, 0.0, 1.0)
        sigma2 = sigma * sigma
        sinp = torch.sin(torch.pi * sigma)
        cosp = torch.cos(torch.pi * sigma)

        parts = [
            torch.ones_like(sigma),
            s1n[..., 0:1], s1n[..., 1:2],
            s2n[..., 0:1], s2n[..., 1:2],
            v1n[..., 0:1], v1n[..., 1:2],
            v2n[..., 0:1], v2n[..., 1:2],
            g_n[..., 0:1], g_n[..., 1:2],
            sigma,
            sigma2,
            sinp,
            cosp,
        ]
        phi = torch.cat(parts, dim=-1)
        return phi

    def critic_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Build critic features φ_c = [1, z (10), quadratic terms (11)].
        """
        state = obs[..., : self.state_dim_3lp]
        goal_rel_pelvis = obs[..., self.state_dim_3lp : self.state_dim_3lp + self.goal_dim]

        X2 = state[..., 0:2]
        x1 = state[..., 2:4]
        X3 = state[..., 4:6]
        X2dot = state[..., 6:8]
        x1dot = state[..., 8:10]

        s1 = x1 - X3
        s2 = X2 - X3
        v1 = x1dot
        v2 = X2dot
        g_rel = goal_rel_pelvis + s1

        L = self.leg_length
        w0 = self.omega0
        s1n = s1 / L
        s2n = s2 / L
        g_n = g_rel / L
        v1n = v1 / (L * w0)
        v2n = v2 / (L * w0)

        z_parts = [
            s1n[..., 0:1], s1n[..., 1:2],
            s2n[..., 0:1], s2n[..., 1:2],
            v1n[..., 0:1], v1n[..., 1:2],
            v2n[..., 0:1], v2n[..., 1:2],
            g_n[..., 0:1], g_n[..., 1:2],
        ]
        z = torch.cat(z_parts, dim=-1)

        q1 = (s1n ** 2).sum(dim=-1, keepdim=True)
        q2 = (s2n ** 2).sum(dim=-1, keepdim=True)
        q3 = (v1n ** 2).sum(dim=-1, keepdim=True)
        q4 = (v2n ** 2).sum(dim=-1, keepdim=True)
        q5 = (g_n ** 2).sum(dim=-1, keepdim=True)
        q6 = (s1n * v1n).sum(dim=-1, keepdim=True)
        q7 = (s2n * v2n).sum(dim=-1, keepdim=True)
        q8 = (s1n * g_n).sum(dim=-1, keepdim=True)
        q9 = (s2n * g_n).sum(dim=-1, keepdim=True)
        q10 = (v1n * g_n).sum(dim=-1, keepdim=True)
        q11 = (v2n * g_n).sum(dim=-1, keepdim=True)

        phi_c = torch.cat(
            [
                torch.ones_like(q1),
                z,
                q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11,
            ],
            dim=-1,
        )
        return phi_c


class GoalWalkEncoder(nn.Module):
    """
    Encoder for the 14-D folded, normalized obs from ThreeLPGoalWalkEnv.
    Actor features match the structured 15-D vector; critic features include quadratic terms.
    """

    def __init__(self, actor_basis_dim: int = 15, critic_basis_dim: int = 22):
        super().__init__()
        self.actor_basis_dim = actor_basis_dim
        self.critic_basis_dim = critic_basis_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs layout:
        # [0] r_pelvis_x, [1] r_pelvis_y,
        # [2] r_swing_x,  [3] r_swing_y,
        # [4] v_pelvis_x, [5] v_pelvis_y,
        # [6] v_swing_x,  [7] v_swing_y,
        # [8] g_rel_x,    [9] g_rel_y,
        # [10] g_dist,    [11] g_heading,
        # [12] phase,     [13] leg_flag
        sigma = obs[..., 12:13]
        sigma2 = sigma * sigma
        sinp = torch.sin(torch.pi * sigma)
        cosp = torch.cos(torch.pi * sigma)
        phi = torch.cat(
            [
                torch.ones_like(sigma),
                obs[..., 0:1], obs[..., 1:2],  # s1
                obs[..., 2:3], obs[..., 3:4],  # s2
                obs[..., 4:5], obs[..., 5:6],  # v1
                obs[..., 6:7], obs[..., 7:8],  # v2
                obs[..., 8:9], obs[..., 9:10],  # g_rel
                sigma,
                sigma2,
                sinp,
                cosp,
            ],
            dim=-1,
        )
        return phi

    def critic_features(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.cat(
            [
                obs[..., 0:1], obs[..., 1:2],  # s1
                obs[..., 2:3], obs[..., 3:4],  # s2
                obs[..., 4:5], obs[..., 5:6],  # v1
                obs[..., 6:7], obs[..., 7:8],  # v2
                obs[..., 8:9], obs[..., 9:10],  # g_rel
            ],
            dim=-1,
        )
        s1 = z[..., 0:2]
        s2 = z[..., 2:4]
        v1 = z[..., 4:6]
        v2 = z[..., 6:8]
        g = z[..., 8:10]
        q1 = (s1 ** 2).sum(dim=-1, keepdim=True)
        q2 = (s2 ** 2).sum(dim=-1, keepdim=True)
        q3 = (v1 ** 2).sum(dim=-1, keepdim=True)
        q4 = (v2 ** 2).sum(dim=-1, keepdim=True)
        q5 = (g ** 2).sum(dim=-1, keepdim=True)
        q6 = (s1 * v1).sum(dim=-1, keepdim=True)
        q7 = (s2 * v2).sum(dim=-1, keepdim=True)
        q8 = (s1 * g).sum(dim=-1, keepdim=True)
        q9 = (s2 * g).sum(dim=-1, keepdim=True)
        q10 = (v1 * g).sum(dim=-1, keepdim=True)
        q11 = (v2 * g).sum(dim=-1, keepdim=True)
        phi_c = torch.cat(
            [
                torch.ones_like(q1),
                z,
                q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11,
            ],
            dim=-1,
        )
        return phi_c


class VelWalkEncoder(nn.Module):
    """
    Encoder for velocity-tracking env (obs 10D: r_pel(2), r_sw(2), v_pel(2), v_sw(2), v_cmd(2)).
    Actor features (16D) and critic features (45D) per the spec.
    """

    def __init__(self, actor_basis_dim: int = 16, critic_basis_dim: int = 45):
        super().__init__()
        self.actor_basis_dim = actor_basis_dim
        self.critic_basis_dim = critic_basis_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        r_px, r_py = obs[..., 0:1], obs[..., 1:2]
        r_sx, r_sy = obs[..., 2:3], obs[..., 3:4]
        v_px, v_py = obs[..., 4:5], obs[..., 5:6]
        v_sx, v_sy = obs[..., 6:7], obs[..., 7:8]
        v_cmdx, v_cmdy = obs[..., 8:9], obs[..., 9:10]
        e_vx = v_px - v_cmdx
        e_vy = v_py - v_cmdy
        T = 0.7  # default step duration guess; could be passed in obs if available
        phi = torch.cat(
            [
                torch.ones_like(r_px),
                e_vx,
                e_vy,
                r_px,
                r_py,
                r_sx,
                r_sy,
                v_sx,
                v_sy,
                v_cmdx,
                v_cmdy,
                T * e_vx,
                T * e_vy,
                r_px / T,
                T * v_px,
                T * v_py,
            ],
            dim=-1,
        )
        return phi

    def critic_features(self, obs: torch.Tensor) -> torch.Tensor:
        # z = [e_vx, e_vy, r_px, r_py, r_sx, r_sy, v_sx, v_sy]
        r_px, r_py = obs[..., 0:1], obs[..., 1:2]
        r_sx, r_sy = obs[..., 2:3], obs[..., 3:4]
        v_px, v_py = obs[..., 4:5], obs[..., 5:6]
        v_sx, v_sy = obs[..., 6:7], obs[..., 7:8]
        v_cmdx, v_cmdy = obs[..., 8:9], obs[..., 9:10]
        e_vx = v_px - v_cmdx
        e_vy = v_py - v_cmdy
        z = torch.cat([e_vx, e_vy, r_px, r_py, r_sx, r_sy, v_sx, v_sy], dim=-1)
        batch, n = z.shape
        terms = [torch.ones(batch, 1, device=z.device, dtype=z.dtype), z]
        quad = []
        for i in range(n):
            for j in range(i, n):
                quad.append((z[:, i] * z[:, j]).unsqueeze(-1))
        quad = torch.cat(quad, dim=-1)
        psi = torch.cat(terms + [quad], dim=-1)
        return psi


class LinearBasisActor(nn.Module):
    """
    Actor: a linear map over basis features.

    a = W φ + b
    with Gaussian exploration: a ~ N(μ, Σ)
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.encoder_type == "goal_walk":
            self.encoder = GoalWalkEncoder(actor_basis_dim=cfg.actor_basis_dim, critic_basis_dim=cfg.critic_basis_dim)
        elif cfg.encoder_type == "vel_walk":
            self.encoder = VelWalkEncoder(actor_basis_dim=cfg.actor_basis_dim, critic_basis_dim=cfg.critic_basis_dim)
        else:
            self.encoder = BasisEncoder(
                obs_dim=cfg.obs_dim,
                actor_basis_dim=cfg.actor_basis_dim,
                critic_basis_dim=cfg.critic_basis_dim,
                state_dim_3lp=12,
            )

        self.linear = nn.Linear(cfg.actor_basis_dim, cfg.action_dim)
        self.log_std = nn.Parameter(torch.ones(cfg.action_dim) * cfg.log_std_init)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and log_std of Gaussian action distribution.
        """
        phi = self.encoder(obs)          # [B, basis_dim]
        mean = self.linear(phi)          # [B, action_dim]
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and return (action, log_prob, mean).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, mean


class MLPActor(nn.Module):
    """
    Simple MLP policy with Gaussian head.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        layers = []
        last = cfg.obs_dim
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, cfg.action_dim)
        self.log_std = nn.Parameter(torch.ones(cfg.action_dim) * cfg.log_std_init)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.mean_head(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, mean


class QuadraticCritic(nn.Module):
    """
    Critic: V(φ) = 0.5 * ||L φ||^2 + bᵀ φ + c

    This implements a general PSD quadratic in the basis features.
    """

    def __init__(self, basis_dim: int):
        super().__init__()
        self.basis_dim = basis_dim

        self.L = nn.Linear(basis_dim, basis_dim, bias=False)  # P = Lᵀ L
        self.b = nn.Parameter(torch.zeros(basis_dim))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        phi: [B, basis_dim]
        returns V: [B]
        """
        y = self.L(phi)             # [B, basis_dim]
        quad = 0.5 * (y ** 2).sum(dim=-1)
        lin = (self.b * phi).sum(dim=-1)
        return quad + lin + self.c


class MLPCritic(nn.Module):
    """
    MLP value function.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        layers = []
        last = cfg.obs_dim
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        self.net = nn.Sequential(*layers)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return self.value_head(x).squeeze(-1)
