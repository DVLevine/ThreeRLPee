import unittest
import numpy as np
import torch

from env_3lp import ThreeLPGotoGoalEnv
from train_3lp_ac import RunningNorm


def make_env():
    # Use Python sim for lightweight deterministic tests
    return ThreeLPGotoGoalEnv(use_python_sim=True, max_episode_steps=50)


def test_fall_detection_on_large_velocity():
    env = make_env()
    obs, info = env.reset()
    env.state_3lp = np.zeros_like(env.state_3lp)
    env.state_3lp[6:] = 100.0  # huge velocity
    reward, terminated, truncated, info = env._compute_reward_and_done(obs)
    assert info["fallen"]
    assert terminated
    assert not truncated


def test_reward_velocity_alignment():
    env = make_env()
    obs, info = env.reset()
    env.goal_world = np.array([1.0, 0.0], dtype=np.float32)
    env.state_3lp = np.zeros_like(env.state_3lp)
    env.prev_dist = 1.0
    env.state_3lp[2] = 0.5  # pelvis ahead
    obs_pos = env._compute_observation()
    r_pos, _, _, info_pos = env._compute_reward_and_done(obs_pos)
    env.prev_dist = 1.0
    env.state_3lp[2] = 0.0  # pelvis behind
    obs_neg = env._compute_observation()
    r_neg, _, _, info_neg = env._compute_reward_and_done(obs_neg)
    assert r_pos > r_neg


def test_observation_shapes():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert obs.shape[0] == env.observation_space.shape[0]
    assert isinstance(reward, float)


def test_running_norm_zero_mean():
    rn = RunningNorm(shape=(2,), device="cpu")
    x = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.float32)
    rn.update(x)
    x_norm = rn.normalize(x)
    mean = x_norm.mean(0)
    var = x_norm.var(0, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
