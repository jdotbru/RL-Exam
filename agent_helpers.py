#Unterstützung von KI in Design-Entscheidungen und teilweise bei Codeteilen

import random

import numpy as np
import torch

from env_helpers import calculate_triangle_area


def choose_antagonist_action(prob: float = 0.0) -> int:
    # Einfacher bounded Antagonist: meistens tut er nichts, manchmal stoert er
    # mit einer zufaelligen 27er-Aktion.
    if random.random() < prob:
        return random.randint(0, 26)
    return 0


def compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    values_t = torch.stack(values)
    next_values = torch.cat([values_t[1:], torch.zeros(1, dtype=values_t.dtype)])
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    deltas = rewards_t + gamma * next_values - values_t

    advantages = torch.zeros_like(rewards_t)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    value_targets = advantages + values_t.detach()
    return advantages, value_targets


def episode_area(env) -> float:
    if env.corner1 is None or env.corner2 is None:
        return 0.0
    try:
        return float(calculate_triangle_area(env.startPos, env.corner1, env.corner2))
    except Exception:
        return 0.0


def moving_average(x: list[float], window: int = 25) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    ma = np.convolve(x, kernel, mode="valid")
    pad = np.full(window - 1, ma[0], dtype=np.float32)
    return np.concatenate([pad, ma])


def moving_success_average(x: list[float], success_mask: list[int], window: int = 25) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    mask = np.asarray(success_mask, dtype=np.float32)
    if len(values) == 0:
        return values

    result = np.zeros_like(values, dtype=np.float32)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start:idx + 1]
        window_mask = mask[start:idx + 1] > 0.5
        if np.any(window_mask):
            result[idx] = float(np.mean(window_values[window_mask]))
        else:
            result[idx] = 0.0
    return result


def stagewise_reward_curves(
    reward_history: list[float],
    stage_history: list[int],
    window: int = 25,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rewards = np.asarray(reward_history, dtype=np.float32)
    stages = np.asarray(stage_history, dtype=np.int32)
    for stage in sorted(set(int(s) for s in stages.tolist())):
        idx = np.where(stages == stage)[0]
        if len(idx) == 0:
            continue
        stage_rewards = rewards[idx]
        x = np.arange(1, len(stage_rewards) + 1, dtype=np.int32)
        y = moving_average(stage_rewards.tolist(), window)
        curves[int(stage)] = (x, y)
    return curves


def stagewise_metric_curves(
    values: list[float],
    stage_history: list[int],
    window: int = 25,
    multiplier: float = 1.0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    metric_values = np.asarray(values, dtype=np.float32)
    stages = np.asarray(stage_history, dtype=np.int32)
    for stage in sorted(set(int(s) for s in stages.tolist())):
        idx = np.where(stages == stage)[0]
        if len(idx) == 0:
            continue
        stage_values = metric_values[idx]
        x = np.arange(1, len(stage_values) + 1, dtype=np.int32)
        y = moving_average((stage_values * multiplier).tolist(), window)
        curves[int(stage)] = (x, y)
    return curves


def stagewise_benchmark_curves(
    episode_points: list[int],
    values: list[float],
    benchmark_stage_history: list[int],
    window: int = 5,
    multiplier: float = 1.0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not episode_points:
        return curves

    x_points = np.asarray(episode_points, dtype=np.int32)
    metric_values = np.asarray(values, dtype=np.float32)
    stages = np.asarray(benchmark_stage_history, dtype=np.int32)
    for stage in sorted(set(int(s) for s in stages.tolist())):
        idx = np.where(stages == stage)[0]
        if len(idx) == 0:
            continue
        stage_x = x_points[idx]
        stage_values = metric_values[idx]
        y = moving_average((stage_values * multiplier).tolist(), window)
        curves[int(stage)] = (stage_x, y)
    return curves
