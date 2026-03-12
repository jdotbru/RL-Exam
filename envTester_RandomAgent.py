import numpy as np
from fingerTriangleEnv import FingerTriangleEnv

def evaluate_random(env, episodes=50):
    returns = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0
        ep_len = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_len)

    print("Random mean return:", np.mean(returns))
    print("Random std return:", np.std(returns))
    print("Random mean ep length:", np.mean(lengths))

env = FingerTriangleEnv()
evaluate_random(env)