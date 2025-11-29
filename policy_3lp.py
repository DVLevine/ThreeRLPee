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
    basis_dim: int
    log_std_init: float = -0.5
    hidden_sizes: tuple[int, ...] = (64, 64)


class BasisEncoder(nn.Module):
    """
    Map observation -> basis features φ(obs).

    You will replace the body of forward() with your 3LP-inspired basis:
      - linear state pieces (3LP state, goal_rel, phase)
      - interactions with phase (e.g. φ ⊗ [1, phase])
      - possibly eigenmode-like functions or [1, t, t^2] on step phase, etc.

    Right now it's a placeholder that:
      - splits obs as [state_3lp, goal_rel (2), phase (1)]
      - creates a simple basis with linear terms + phase^2 + interactions.
    """

    def __init__(self, obs_dim: int, basis_dim: int, state_dim_3lp: int, goal_dim: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.basis_dim = basis_dim
        self.state_dim_3lp = state_dim_3lp
        self.goal_dim = goal_dim

        # Optionally put a small linear layer to map a big obs vector down to basis_dim.
        self.linear = nn.Linear(obs_dim, basis_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, obs_dim]
        returns φ: [B, basis_dim]
        """
        # --- Example: simple linear mapping as placeholder ---
        # Replace with explicit 3LP basis design if you want full control.
        phi = self.linear(obs)

        # Optional: add some polynomial/nonlinear features on top
        phase = obs[..., -1:]  # assumes last dim is phase in [0,1]
        phi = torch.cat([phi, phase, phase ** 2], dim=-1)

        # Truncate / project to desired basis_dim if needed
        if phi.shape[-1] > self.basis_dim:
            phi = phi[..., : self.basis_dim]
        elif phi.shape[-1] < self.basis_dim:
            # Zero-pad
            pad = self.basis_dim - phi.shape[-1]
            phi = F.pad(phi, (0, pad))

        return phi


class LinearBasisActor(nn.Module):
    """
    Actor: a linear map over basis features.

    a = W φ + b
    with Gaussian exploration: a ~ N(μ, Σ)
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = BasisEncoder(
            obs_dim=cfg.obs_dim,
            basis_dim=cfg.basis_dim,
            state_dim_3lp=cfg.obs_dim - 3,  # crude guess; adjust or pass explicitly
        )

        self.linear = nn.Linear(cfg.basis_dim, cfg.action_dim)
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
