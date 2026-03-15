import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from fingerTriangleEnv import FingerTriangleEnv


def make_env(use_antagonist: bool = False, seed: int | None = None):
    env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def evaluate_policy_like(env, model=None, episodes: int = 50, random_policy: bool = False):
    returns = []
    lengths = []
    successes = 0
    areas = []
    straightness = []
    extra_corners = []
    phase2_reached = 0

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_len)

        # Erfolg = Environment wurde regulär abgeschlossen
        if terminated:
            successes += 1

        base_env = env.unwrapped
        area = float(info.get("area", 0.0))
        straightness.append(float(info.get("triangle_straightness", 0.0)))
        extra_corners.append(int(info.get("extra_corners", 0)))
        if int(info.get("current_Phase", 0)) >= 2:
            phase2_reached += 1
        areas.append(area)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_len": float(np.mean(lengths)),
        "success_rate": float(successes / episodes),
        "mean_area": float(np.mean(areas)),
        "max_area": float(np.max(areas)),
        "mean_straightness": float(np.mean(straightness)),
        "mean_extra_corners": float(np.mean(extra_corners)),
        "phase2_rate": float(phase2_reached / episodes),
    }


def main():
    TRAIN_STEPS = 30_000
    EVAL_EPISODES = 50
    USE_ANTAGONIST = False

    # Random-Baseline
    random_env = make_env(use_antagonist=USE_ANTAGONIST, seed=0)
    random_stats = evaluate_policy_like(
        random_env,
        model=None,
        episodes=EVAL_EPISODES,
        random_policy=True
    )

    # PPO-Training
    train_env = make_env(use_antagonist=USE_ANTAGONIST, seed=1)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=1,
    )

    model.learn(total_timesteps=TRAIN_STEPS)

    # PPO-Evaluation
    eval_env = make_env(use_antagonist=USE_ANTAGONIST, seed=2)
    ppo_stats = evaluate_policy_like(
        eval_env,
        model=model,
        episodes=EVAL_EPISODES,
        random_policy=False
    )

    print("\n--- RESULTS ---")
    print(
        f"Random: mean_return={random_stats['mean_return']:.3f}  "
        f"std={random_stats['std_return']:.3f}  "
        f"mean_len={random_stats['mean_len']:.1f}  "
        f"success_rate={random_stats['success_rate']:.2%}  "
        f"mean_area={random_stats['mean_area']:.3f}  "
        f"max_area={random_stats['max_area']:.3f}  "
        f"phase2_rate={random_stats['phase2_rate']:.2%}  "
        f"straight={random_stats['mean_straightness']:.3f}  "
        f"extra_corners={random_stats['mean_extra_corners']:.2f}"
    )
    print(
        f"PPO:    mean_return={ppo_stats['mean_return']:.3f}  "
        f"std={ppo_stats['std_return']:.3f}  "
        f"mean_len={ppo_stats['mean_len']:.1f}  "
        f"success_rate={ppo_stats['success_rate']:.2%}  "
        f"mean_area={ppo_stats['mean_area']:.3f}  "
        f"max_area={ppo_stats['max_area']:.3f}  "
        f"phase2_rate={ppo_stats['phase2_rate']:.2%}  "
        f"straight={ppo_stats['mean_straightness']:.3f}  "
        f"extra_corners={ppo_stats['mean_extra_corners']:.2f}"
    )


if __name__ == "__main__":
    main()
