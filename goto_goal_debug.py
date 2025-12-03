import numpy as np
from env_3lp import ThreeLPGotoGoalEnv

env = ThreeLPGotoGoalEnv(use_python_sim=False, debug_log=True)
obs, info = env.reset()
print("Initial dist_to_goal:", np.linalg.norm(info["goal_world"]))
print("Initial obs:", obs)

total_rew = 0.0
for t in range(200):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, rew, terminated, truncated, info = env.step(action)
    total_rew += rew
    print(f"t={t}, rew={rew:.3f}, dist={info['dist_to_goal']:.3f}, fallen={info['fallen']}")
    if terminated or truncated:
        print("Episode ended:", "terminated" if terminated else "truncated")
        break

print("Total reward:", total_rew)
