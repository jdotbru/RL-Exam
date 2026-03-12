import numpy as np
from fingerTriangleEnv import FingerTriangleEnv

env = FingerTriangleEnv()

obs, info = env.reset()
print("Initial obs:", obs)
print("Obs shape:", obs.shape)

for i in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs), f"Ungültige Observation bei Schritt {i}"
    assert isinstance(reward, (int, float, np.floating)), f"Ungültiger Reward bei Schritt {i}"

    if terminated or truncated:
        print(f"Episode beendet bei Schritt {i}, reward={reward}")
        obs, info = env.reset()

print("API-Test bestanden.")