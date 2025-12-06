import numpy as np

from env_high_rate_3lp import ThreeLPHighRateEnv


def debug_step():
    env = ThreeLPHighRateEnv(t_ds=0.1, t_ss=0.6, dt=0.02, seed=42, max_steps=10)
    obs, info = env.reset()

    print("\n--- STEP 1 (Left Stance) ---")
    print(f"support_sign: {env.support_sign}")
    print(f"phase: {env.phase}")
    print(f"obs (first 4): {obs[:4]}")

    action = np.zeros(8, dtype=np.float64)
    swapped = False
    for k in range(20):
        obs, rew, done, trunc, info = env.step(action)
        if env.support_sign < 0:
            swapped = True
            break

    print("\n--- FIRST SWAP ---")
    if not swapped:
        print("Did not reach stance swap within 20 steps.")
    print(f"support_sign: {env.support_sign} (expect -1 when swapped)")
    print(f"phase: {env.phase}")
    print(f"obs (first 4): {obs[:4]}")
    s1y = obs[1]
    s2y = obs[3]
    print(f"  s1y (stance-pelvis y): {s1y:.4f}")
    print(f"  s2y (swing-pelvis y): {s2y:.4f}")
    if s2y > 0:
        print("  !!! ALERT: swing leg appears +Y (inward) in canonical observation !!!")
    else:
        print("  Observation swing is on canonical (outward) side.")

    try:
        q_world = env.sim.get_state_world().q
        print(f"\nWorld swing y (q[1]): {q_world[1]:.4f}")
    except Exception as e:
        print(f"(Could not access sim.get_state_world: {e})")


if __name__ == "__main__":
    debug_step()
