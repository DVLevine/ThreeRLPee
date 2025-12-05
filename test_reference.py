import numpy as np

from env_high_rate_3lp import ThreeLPHighRateEnv


def test_reference_stability(dt=0.005, max_steps=500):
    print("=== Testing Reference Gait Stability (Zero Action) ===")

    env = ThreeLPHighRateEnv(
        dt=dt,
        t_ds=0.1,
        t_ss=0.6,
        reset_noise_std=0.0,
        random_phase=False,  # start at phase 0 to follow the reference exactly
        alive_bonus=0.0,
        action_clip=100.0,
    )

    obs, _ = env.reset()
    print(f"Target Speed: {env.v_cmd:.2f} m/s | dt={env.dt:.4f}s")

    done = False
    step_count = 0
    total_reward = 0.0

    while not done and step_count < max_steps:
        # Zero action means the environment replays the cached reference torques.
        action = np.zeros(env.action_dim, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward
        step_count += 1

        if step_count % 50 == 0:
            s1x = float(obs[0])  # sagittal position
            print(f"Step {step_count}: phase={info['phase']}, s1x={s1x:.3f}")

    env.close()

    if step_count >= max_steps:
        print("\n[SUCCESS] Reference gait is stable open-loop.")
    else:
        print(f"\n[FAILURE] Reference gait fell at step {step_count}.")
        print("Likely causes: integration dt too large or bad reference parameters.")
    print(f"Total reward accrued: {total_reward:.3f}")


if __name__ == "__main__":
    test_reference_stability()
