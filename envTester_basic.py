import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from fingerTriangleEnv import FingerTriangleEnv  # ggf. Pfad/Name anpassen

env = FingerTriangleEnv()
check_env(env, skip_render_check=True)
print("check_env: OK")