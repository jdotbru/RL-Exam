import random
import traceback
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from fingerTriangleEnv import FingerTriangleEnv


# -------------------------------------------------
# Echter Actor-Critic: gemeinsames Netz mit 2 Köpfen
# -------------------------------------------------
class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 27):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return logits, value


@dataclass
class EpisodeResult:
    episode: int
    reward: float
    area: float
    success: bool
    positions: np.ndarray
    start: np.ndarray
    corner1: np.ndarray | None
    corner2: np.ndarray | None
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    final_phase: int = 0
    found_corner1: bool = False
    found_corner2: bool = False
    distance_to_start: float = 0.0
    steps: int = 0
    mean_line_deviation: float = 0.0
    triangle_straightness: float = 0.0
    extra_corners: int = 0
    curriculum_stage: int = 0
    good_triangle: bool = False
    origin: str = "training"
    shaping_reward: float = 0.0
    terminal_reward: float = 0.0
    terminal_area_bonus: float = 0.0
    terminal_good_bonus: float = 0.0
    best_form_snapshot: dict | None = None


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def choose_antagonist_action(prob: float = 0.0) -> int:
    """
    Einfacher bounded Antagonist:
    - meistens tut er nichts (Aktion 0)
    - manchmal stört er mit einer kleinen zufälligen 27er-Aktion
    """
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


def episode_area(env: FingerTriangleEnv) -> float:
    if env.corner1 is None or env.corner2 is None:
        return 0.0
    try:
        return float(env.calculateTriangleArea(env.startPos, env.corner1, env.corner2))
    except Exception:
        return 0.0


# -------------------------------------------------
# Eine Episode sammeln
# -------------------------------------------------
def run_episode(
    env: FingerTriangleEnv,
    model: ActorCriticNet,
    gamma: float,
    gae_lambda: float,
    antagonist_prob: float,
    deterministic: bool = False,
    sampling_temperature: float = 1.0,
):
    obs, _ = env.reset()

    observations: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[float] = []

    done = False
    truncated = False
    final_info = {
        "current_Phase": 0,
        "has_corner1": False,
        "has_corner2": False,
        "distance_to_start": 0.0,
        "step_ctr": 0,
        "mean_line_deviation": 0.0,
        "triangle_straightness": 0.0,
        "extra_corners": 0,
        "good_triangle": False,
        "shaping_reward": 0.0,
        "terminal_reward": 0.0,
        "terminal_area_bonus": 0.0,
        "terminal_good_bonus": 0.0,
    }

    while not (done or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = model(obs_t)
        if deterministic:
            dist = torch.distributions.Categorical(logits=logits)
            protagonist_action = torch.argmax(logits, dim=-1)
        else:
            temperature = max(0.05, float(sampling_temperature))
            dist = torch.distributions.Categorical(logits=logits / temperature)
            protagonist_action = dist.sample()

        action = {
            "protagonist": int(protagonist_action.item()),
            "antagonist": choose_antagonist_action(antagonist_prob),
        }

        next_obs, reward, done, truncated, info = env.step(action)

        observations.append(np.array(obs, dtype=np.float32))
        actions.append(int(protagonist_action.item()))
        log_probs.append(dist.log_prob(protagonist_action).squeeze())
        values.append(value.squeeze())
        entropies.append(dist.entropy().squeeze())
        rewards.append(float(reward))

        obs = next_obs
        final_info = info

    result = EpisodeResult(
        episode=0,
        reward=float(sum(rewards)),
        area=episode_area(env),
        success=bool(done),
        positions=np.array(env.positionSaver, dtype=np.float32),
        start=np.array(env.startPos, dtype=np.float32),
        corner1=None if env.corner1 is None else np.array(env.corner1, dtype=np.float32),
        corner2=None if env.corner2 is None else np.array(env.corner2, dtype=np.float32),
        final_phase=int(final_info["current_Phase"]),
        found_corner1=bool(final_info["has_corner1"]),
        found_corner2=bool(final_info["has_corner2"]),
        distance_to_start=float(final_info["distance_to_start"]),
        steps=int(final_info["step_ctr"]),
        mean_line_deviation=float(final_info["mean_line_deviation"]),
        triangle_straightness=float(final_info["triangle_straightness"]),
        extra_corners=int(final_info["extra_corners"]),
        good_triangle=bool(final_info["good_triangle"]),
        origin="training",
        shaping_reward=float(final_info["shaping_reward"]),
        terminal_reward=float(final_info["terminal_reward"]),
        terminal_area_bonus=float(final_info["terminal_area_bonus"]),
        terminal_good_bonus=float(final_info["terminal_good_bonus"]),
        best_form_snapshot=env.bestFormSnapshot,
    )

    observations_t = torch.tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.int64)
    log_probs_t = torch.stack(log_probs).detach()
    entropies_t = torch.stack(entropies)
    advantages_t, value_targets_t = compute_gae(
        rewards=rewards,
        values=values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    values_t = torch.stack(values)

    return result, observations_t, actions_t, log_probs_t, values_t, advantages_t, value_targets_t, entropies_t


def evaluate_deterministic_policy(
    env: FingerTriangleEnv,
    model: ActorCriticNet,
    gamma: float,
    gae_lambda: float,
    episodes: int = 20,
) -> dict[str, float]:
    rewards = []
    areas = []
    successes = []

    with torch.no_grad():
        for _ in range(episodes):
            result, _, _, _, _, _, _, _ = run_episode(
                env=env,
                model=model,
                gamma=gamma,
                gae_lambda=gae_lambda,
                antagonist_prob=0.0,
                deterministic=True,
            )
            rewards.append(result.reward)
            areas.append(result.area)
            successes.append(int(result.success))

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_area": float(np.mean(areas)),
        "success_rate": float(np.mean(successes) * 100.0),
    }


def episode_quality_score(result: EpisodeResult) -> float:
    score = (
        220.0 * result.area
        + 45.0 * result.triangle_straightness
        - 10.0 * result.extra_corners
        - 35.0 * result.distance_to_start
    )
    if result.success:
        score += 160.0
    if result.good_triangle:
        score += 80.0
    return float(score)


def clean_triangle_like(result: EpisodeResult) -> bool:
    return bool(
        result.success
        and result.distance_to_start <= 0.60
        and result.triangle_straightness >= 0.60
        and result.extra_corners <= 1
    )


def visually_closed(result: EpisodeResult) -> bool:
    return bool(
        result.success
        and result.distance_to_start <= 0.60
    )


def visually_closed_score(result: EpisodeResult) -> tuple[float, float, float, float]:
    return (
        float(visually_closed(result)),
        -result.distance_to_start,
        result.area,
        result.triangle_straightness,
    )


def best_form_quality_score(result: EpisodeResult) -> tuple[float, float, float, float]:
    snapshot = result.best_form_snapshot
    if snapshot is None:
        return (
            float(result.good_triangle),
            result.triangle_straightness,
            -result.distance_to_start,
            result.area,
        )
    return (
        float(snapshot["success_like"]),
        float(snapshot["triangle_straightness"]),
        -float(snapshot["distance_to_start"]),
        float(snapshot["area"]),
    )


def best_form_variant(result: EpisodeResult) -> EpisodeResult:
    snapshot = result.best_form_snapshot
    if snapshot is None:
        return result
    return replace(
        result,
        positions=np.array(snapshot["positions"], dtype=np.float32),
        area=float(snapshot["area"]),
        success=bool(snapshot["success_like"]),
        corner1=None if snapshot["corner1"] is None else np.array(snapshot["corner1"], dtype=np.float32),
        corner2=None if snapshot["corner2"] is None else np.array(snapshot["corner2"], dtype=np.float32),
        distance_to_start=float(snapshot["distance_to_start"]),
        mean_line_deviation=float(snapshot["mean_line_deviation"]),
        triangle_straightness=float(snapshot["triangle_straightness"]),
        extra_corners=int(snapshot["extra_corners"]),
        curriculum_stage=int(snapshot["curriculum_stage"]),
        origin="best_form_snapshot",
    )


def build_showcase_episodes(candidates: list[tuple[str, EpisodeResult]], limit: int = 4) -> list[tuple[str, EpisodeResult]]:
    selected: list[tuple[str, EpisodeResult]] = []
    seen_keys: set[tuple[int, str]] = set()

    for label, result in candidates:
        if result is None:
            continue
        key = (int(result.episode), str(result.origin))
        if key in seen_keys:
            continue
        selected.append((label, result))
        seen_keys.add(key)
        if len(selected) >= limit:
            break

    return selected


def build_success_showcases(results: list[EpisodeResult], limit: int = 4) -> list[tuple[str, EpisodeResult]]:
    successes = [result for result in results if result.success]
    if not successes:
        return []

    good_successes = [result for result in successes if result.good_triangle]
    visually_closed_successes = [result for result in successes if visually_closed(result)]
    visually_closed_good = [result for result in visually_closed_successes if result.good_triangle]
    score_pool = visually_closed_good or visually_closed_successes or good_successes or successes
    shape_pool = good_successes or successes

    candidates = [
        ("Final: Bestes Erfolgs-Ep.", max(score_pool, key=episode_quality_score)),
        ("Final: Groesste Erfolgs-Flaeche", max(score_pool, key=lambda result: (visually_closed_score(result), result.area))),
        ("Final: Beste Erfolgs-Form", best_form_variant(max(shape_pool, key=best_form_quality_score))),
        (
            "Final: Sauberster Erfolg",
            min(score_pool, key=lambda result: (result.distance_to_start, result.extra_corners, -result.triangle_straightness, -result.area)),
        ),
    ]
    return build_showcase_episodes(candidates, limit=limit)


def build_sampler_showcases(results: list[EpisodeResult], limit: int = 4) -> list[tuple[str, EpisodeResult]]:
    successful = [result for result in results if result.success]
    if not successful:
        phase2_results = [result for result in results if result.final_phase >= 2]
        pool = phase2_results if phase2_results else results
        candidates = [
            ("Sampler: Bestes Gesamt-Ep.", max(pool, key=episode_quality_score)),
            ("Sampler: Beste Form", best_form_variant(max(pool, key=best_form_quality_score))),
            ("Sampler: Groesste Flaeche", max(pool, key=lambda result: result.area)),
            ("Sampler: Naechster Abschluss", min(pool, key=lambda result: (result.distance_to_start, result.extra_corners, -result.triangle_straightness, -result.area))),
        ]
        return build_showcase_episodes(candidates, limit=limit)

    good_successes = [result for result in successful if result.good_triangle]
    visually_closed_successes = [result for result in successful if visually_closed(result)]
    visually_closed_good = [result for result in visually_closed_successes if result.good_triangle]
    score_pool = visually_closed_good or visually_closed_successes or good_successes or successful
    shape_pool = good_successes or successful

    candidates = [
        ("Sampler: Bestes Gesamt-Ep.", max(score_pool, key=episode_quality_score)),
        ("Sampler: Sauberster Abschluss", min(score_pool, key=lambda result: (result.distance_to_start, result.extra_corners, -result.triangle_straightness, -result.area))),
        ("Sampler: Beste Form", best_form_variant(max(shape_pool, key=best_form_quality_score))),
        ("Sampler: Groesste Flaeche", max(score_pool, key=lambda result: (visually_closed_score(result), result.area))),
    ]
    return build_showcase_episodes(candidates, limit=limit)


def build_progress_showcases(results: list[EpisodeResult]) -> list[tuple[str, EpisodeResult]]:
    if not results:
        return []

    selected: list[tuple[str, EpisodeResult]] = []
    stage_labels = {
        0: "Beste Form Stage 0",
        1: "Beste Form Stage 1",
        2: "Beste Form Stage 2",
        3: "Beste Form Stage 3",
    }

    for stage in [0, 1, 2, 3]:
        stage_results = [result for result in results if result.curriculum_stage == stage]
        if not stage_results:
            continue
        stage_successes = [result for result in stage_results if result.success]
        source = stage_successes if stage_successes else stage_results
        candidate = best_form_variant(max(source, key=best_form_quality_score))
        selected.append((stage_labels[stage], candidate))

    return build_showcase_episodes(selected, limit=4)


def collect_evaluation_rollouts(
    env: FingerTriangleEnv,
    model: ActorCriticNet,
    gamma: float,
    gae_lambda: float,
    episodes: int = 60,
    sampling_temperature: float = 1.0,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []

    with torch.no_grad():
        for idx in range(1, episodes + 1):
            result, _, _, _, _, _, _, _ = run_episode(
                env=env,
                model=model,
                gamma=gamma,
                gae_lambda=gae_lambda,
                antagonist_prob=0.0,
                deterministic=False,
                sampling_temperature=sampling_temperature,
            )
            result.episode = idx
            result.origin = "final_eval"
            results.append(result)

    return results


def collect_temperature_sweep_rollouts(
    env: FingerTriangleEnv,
    model: ActorCriticNet,
    gamma: float,
    gae_lambda: float,
    temperatures: list[float],
    episodes_per_temperature: int,
) -> tuple[list[EpisodeResult], list[dict[str, float]]]:
    all_results: list[EpisodeResult] = []
    summaries: list[dict[str, float]] = []

    next_episode = 1
    for temperature in temperatures:
        temp_results = collect_evaluation_rollouts(
            env=env,
            model=model,
            gamma=gamma,
            gae_lambda=gae_lambda,
            episodes=episodes_per_temperature,
            sampling_temperature=temperature,
        )
        for result in temp_results:
            result.episode = next_episode
            next_episode += 1
        all_results.extend(temp_results)

        successful = [result for result in temp_results if result.success]
        good_successes = [result for result in successful if result.good_triangle]
        summaries.append(
            {
                "temperature": float(temperature),
                "success_rate": 100.0 * len(successful) / max(1, len(temp_results)),
                "good_triangle_rate": 100.0 * len(good_successes) / max(1, len(temp_results)),
                "best_area": max((result.area for result in temp_results), default=0.0),
                "best_straightness": max((result.triangle_straightness for result in temp_results), default=0.0),
            }
        )

    return all_results, summaries


def checkpoint_score_tuple(results: list[EpisodeResult]) -> tuple[float, float, float, float, float]:
    if not results:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    successes = [result for result in results if result.success]
    good_successes = [result for result in successes if result.good_triangle]
    clean_successes = [result for result in successes if clean_triangle_like(result)]
    success_rate = len(successes) / len(results)
    good_rate = len(good_successes) / len(results)
    clean_rate = len(clean_successes) / len(results)
    mean_clean_area = float(np.mean([result.area for result in clean_successes])) if clean_successes else 0.0
    best_success_area = max((result.area for result in successes), default=0.0)
    return (clean_rate, good_rate, mean_clean_area, success_rate, best_success_area)


# -----------------------------
# Visualisierung
# -----------------------------
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


def stagewise_reward_curves(reward_history: list[float], stage_history: list[int], window: int = 25) -> dict[int, tuple[np.ndarray, np.ndarray]]:
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


def plot_episode(ax, label: str, result: EpisodeResult):
    if result.origin == "final_eval":
        id_label = "Rollout"
    elif result.origin == "best_form_snapshot":
        id_label = "Snapshot"
    else:
        id_label = "Ep"
    ax.set_title(
        f"{label}\n"
        f"{id_label} {result.episode} | Stage {result.curriculum_stage}\n"
        f"A={result.area:.2f}, Straight={result.triangle_straightness:.2f}, dStart={result.distance_to_start:.2f}, Extra={result.extra_corners}, Erfolg={result.success}"
    )

    if result.positions is not None and len(result.positions) > 0:
        rel = result.positions - result.start
        ax.plot(rel[:, 0], rel[:, 1], linewidth=2)

        ax.scatter(0.0, 0.0, s=60, marker="s", label="Start")
        ax.scatter(rel[-1, 0], rel[-1, 1], s=40, marker="o", label="Ende")

        if result.corner1 is not None:
            c1 = result.corner1 - result.start
            ax.scatter(c1[0], c1[1], s=60, marker="^", label="Ecke 1")

        if result.corner2 is not None:
            c2 = result.corner2 - result.start
            ax.scatter(c2[0], c2[1], s=60, marker="x", label="Ecke 2")

            if result.corner1 is not None:
                c1 = result.corner1 - result.start
                triangle = np.vstack([
                    np.array([0.0, 0.0], dtype=np.float32),
                    c1,
                    c2,
                    np.array([0.0, 0.0], dtype=np.float32),
                ])
                ax.plot(triangle[:, 0], triangle[:, 1], linestyle="--", linewidth=1.5, alpha=0.8, label="Dreieck")

    ax.axhline(0, linewidth=0.6, alpha=0.4)
    ax.axvline(0, linewidth=0.6, alpha=0.4)
    ax.set_xlabel("x relativ zum Start")
    ax.set_ylabel("y relativ zum Start")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def add_stage_background(ax, stage_history: list[int]) -> None:
    if not stage_history:
        return

    stage_colors = {
        0: "#eaf4d3",
        1: "#d8f3dc",
        2: "#dceaf7",
        3: "#fde2e4",
    }

    start_idx = 0
    current_stage = stage_history[0]
    for idx, stage in enumerate(stage_history[1:], start=1):
        if stage != current_stage:
            ax.axvspan(start_idx, idx, color=stage_colors.get(current_stage, "#f0f0f0"), alpha=0.18)
            start_idx = idx
            current_stage = stage
    ax.axvspan(start_idx, len(stage_history), color=stage_colors.get(current_stage, "#f0f0f0"), alpha=0.18)


def plot_results(
    reward_history,
    area_history,
    success_history,
    successful_area_history,
    straightness_history,
    extra_corner_history,
    good_triangle_history,
    actor_loss_history,
    critic_loss_history,
    shaping_reward_history,
    terminal_reward_history,
    stage_history,
    showcases,
    progress_showcases,
):
    reward_ma = moving_average(reward_history, 25)
    area_ma = moving_average(area_history, 25)
    success_ma = moving_average(success_history, 25) * 100.0
    successful_area_ma = moving_success_average(successful_area_history, success_history, 25)
    straightness_ma = moving_average(straightness_history, 25)
    extra_corner_ma = moving_average(extra_corner_history, 25)
    good_triangle_ma = moving_average(good_triangle_history, 25) * 100.0
    actor_loss_ma = moving_average(actor_loss_history, 25)
    critic_loss_ma = moving_average(critic_loss_history, 25)
    shaping_reward_ma = moving_average(shaping_reward_history, 25)
    terminal_reward_ma = moving_average(terminal_reward_history, 25)
    stage_reward_curves = stagewise_reward_curves(reward_history, stage_history, 25)

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle("Curriculum-Training und Entwicklung der Dreiecksform", fontsize=18)

    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(reward_history, alpha=0.25, label="Reward pro Episode")
    ax1.plot(reward_ma, linewidth=2.5, label="Moving Average (25)")
    add_stage_background(ax1, stage_history)
    ax1.set_title("Reward über die Zeit")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(area_history, alpha=0.25, label="Fläche pro Episode")
    ax2.plot(area_ma, linewidth=2.5, label="Moving Average (25)")
    add_stage_background(ax2, stage_history)
    ax2.set_title("Dreiecksfläche über die Zeit")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Fläche")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(success_ma, linewidth=2.5)
    add_stage_background(ax3, stage_history)
    ax3.set_title("Success-Rate (Moving Average 25)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Erfolgsquote in %")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(straightness_history, alpha=0.25, label="Straightness")
    ax4.plot(straightness_ma, linewidth=2.5, label="Straightness MA")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(extra_corner_history, alpha=0.2, color="tab:red", label="Extra Corners")
    ax4_twin.plot(extra_corner_ma, linewidth=2.0, color="tab:red", label="Extra Corners MA")
    add_stage_background(ax4, stage_history)
    ax4.set_title("Formqualität")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Straightness")
    ax4_twin.set_ylabel("Extra Corners")
    ax4.grid(True, alpha=0.3)
    lines_left, labels_left = ax4.get_legend_handles_labels()
    lines_right, labels_right = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="upper right")

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(good_triangle_ma, linewidth=2.5, color="tab:green")
    add_stage_background(ax5, stage_history)
    ax5.set_title("Good-Triangle-Rate (MA 25)")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Rate in %")
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 1])
    stage_colors = {
        0: "tab:green",
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:red",
    }
    for stage, (x_vals, y_vals) in stage_reward_curves.items():
        ax6.plot(x_vals, y_vals, linewidth=2.2, color=stage_colors.get(stage, None), label=f"Stage {stage}")
    ax6.set_title("Reward Je Stage (MA 25)")
    ax6.set_xlabel("Episode innerhalb der Stage")
    ax6.set_ylabel("Reward")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(critic_loss_history, alpha=0.25, label="Critic Loss")
    ax7.plot(critic_loss_ma, linewidth=2.5, label="Critic Loss MA")
    add_stage_background(ax7, stage_history)
    ax7.set_title("Critic Loss")
    ax7.set_xlabel("Episode")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(successful_area_ma, linewidth=2.5, color="tab:orange", label="Erfolgsflaeche MA")
    ax8.plot(area_history, alpha=0.08, color="tab:orange")
    ax8_twin = ax8.twinx()
    ax8_twin.plot(shaping_reward_ma, linewidth=1.8, color="tab:blue", alpha=0.8, label="Shape Reward MA")
    ax8_twin.plot(terminal_reward_ma, linewidth=1.8, color="tab:purple", alpha=0.8, label="Terminal Reward MA")
    add_stage_background(ax8, stage_history)
    ax8.set_title("Erfolgsflaeche und Reward-Anteile")
    ax8.set_xlabel("Episode")
    ax8.set_ylabel("Mean Area bei Erfolgen")
    ax8_twin.set_ylabel("Reward-Anteile")
    ax8.grid(True, alpha=0.3)
    lines_left, labels_left = ax8.get_legend_handles_labels()
    lines_right, labels_right = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="upper right")

    progress_handles = []
    progress_labels = []
    if progress_showcases:
        for idx, (label, result) in enumerate(progress_showcases[:4]):
            ax = fig.add_subplot(gs[2, idx])
            plot_episode(ax, label, result)
            if not progress_handles:
                progress_handles, progress_labels = ax.get_legend_handles_labels()

    showcase_handles = []
    showcase_labels = []
    if showcases:
        for idx, (label, result) in enumerate(showcases[:4]):
            ax = fig.add_subplot(gs[3, idx])
            plot_episode(ax, label, result)
            if not showcase_handles:
                showcase_handles, showcase_labels = ax.get_legend_handles_labels()
    elif not progress_showcases:
        ax = fig.add_subplot(gs[3, :])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Keine erfolgreichen Final-Stage-Rollouts gefunden.",
            ha="center",
            va="center",
            fontsize=16,
        )

    combined_handles = progress_handles or showcase_handles
    combined_labels = progress_labels or showcase_labels
    if combined_handles:
        fig.legend(combined_handles, combined_labels, loc="lower center", ncol=4)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()


# -----------------------------
# Hauptlauf
# -----------------------------
def main():
    # Reproduzierbarkeit
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    quick_test = True

    # Hyperparameter
    episodes = 1200 if quick_test else 3000
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    critic_weight = 0.5
    entropy_weight_start = 0.002
    entropy_weight_end = 0.0001
    update_batch_episodes = 8
    ppo_epochs = 4
    ppo_minibatch_size = 256
    ppo_clip_epsilon = 0.2
    antagonist_prob = 0.0
    use_antagonist = True
    final_eval_temperatures = [0.75]
    final_eval_episodes_per_temperature = 80 if quick_test else 360
    checkpoint_eval_every = 200 if quick_test else 100
    checkpoint_eval_episodes = 16 if quick_test else 48
    periodic_eval_episodes = 5 if quick_test else 15
    best_checkpoint_path = "best_triangle_checkpoint.pt"
    final_summary_path = "final_eval_summary.txt"

    env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
    eval_env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
    obs_dim = int(np.prod(env.observation_space.shape))

    model = ActorCriticNet(obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reward_history: list[float] = []
    area_history: list[float] = []
    success_history: list[int] = []
    successful_area_history: list[float] = []
    actor_loss_history: list[float] = []
    critic_loss_history: list[float] = []
    corner1_history: list[int] = []
    corner2_history: list[int] = []
    phase2_history: list[int] = []
    straightness_history: list[float] = []
    extra_corner_history: list[int] = []
    good_triangle_history: list[int] = []
    shaping_reward_history: list[float] = []
    terminal_reward_history: list[float] = []
    terminal_area_bonus_history: list[float] = []
    terminal_good_bonus_history: list[float] = []
    stage_history: list[int] = []
    training_results: list[EpisodeResult] = []
    best_checkpoint_state: dict[str, torch.Tensor] | None = None
    best_checkpoint_score = (-1.0, -1.0, -1.0, -1.0, -1.0)
    best_checkpoint_episode = 0

    if quick_test:
        stage_endpoints = (320, 680, 980, episodes)
    else:
        stage_endpoints = (800, 1800, 2600, episodes)

    def curriculum_stage_for_episode(episode: int) -> int:
        if episode <= stage_endpoints[0]:
            return 0
        if episode <= stage_endpoints[1]:
            return 1
        if episode <= stage_endpoints[2]:
            return 2
        return 3

    print("Starte Actor-Critic-Lauf ...")
    print("Der Actor lernt die Policy, der Critic lernt den Zustandswert V(s).")
    print("Optimiert wird: PPO-Clip-Loss + Critic-Loss - Entropiebonus.\n")
    print(f"Modus: {'Quick Test' if quick_test else 'Full Run'}\n")

    episodes_since_update = 0
    batch_obs: list[torch.Tensor] = []
    batch_actions: list[torch.Tensor] = []
    batch_old_log_probs: list[torch.Tensor] = []
    batch_advantages: list[torch.Tensor] = []
    batch_value_targets: list[torch.Tensor] = []

    for ep in range(1, episodes + 1):
        stage = curriculum_stage_for_episode(ep)
        env.set_curriculum_stage(stage)
        eval_env.set_curriculum_stage(stage)

        result, observations, actions, old_log_probs, values, advantages, value_targets, entropies = run_episode(
            env=env,
            model=model,
            gamma=gamma,
            gae_lambda=gae_lambda,
            antagonist_prob=antagonist_prob,
        )
        result.episode = ep
        result.curriculum_stage = stage

        logits, predicted_values = model(observations)
        dist = torch.distributions.Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions)
        ratio = torch.exp(current_log_probs - old_log_probs)
        unclipped = ratio * advantages.detach()
        clipped = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages.detach()
        actor_loss = -torch.min(unclipped, clipped).mean()
        critic_loss = F.mse_loss(predicted_values, value_targets)
        entropy_bonus = entropies.mean()
        progress = (ep - 1) / max(1, episodes - 1)
        entropy_weight = entropy_weight_start + progress * (entropy_weight_end - entropy_weight_start)

        batch_obs.append(observations.detach())
        batch_actions.append(actions.detach())
        batch_old_log_probs.append(old_log_probs.detach())
        batch_advantages.append(advantages.detach())
        batch_value_targets.append(value_targets.detach())
        episodes_since_update += 1

        if episodes_since_update >= update_batch_episodes or ep == episodes:
            obs_batch = torch.cat(batch_obs, dim=0)
            action_batch = torch.cat(batch_actions, dim=0)
            old_log_prob_batch = torch.cat(batch_old_log_probs, dim=0)
            advantage_batch = torch.cat(batch_advantages, dim=0)
            value_target_batch = torch.cat(batch_value_targets, dim=0)

            num_samples = obs_batch.shape[0]
            minibatch_size = min(ppo_minibatch_size, num_samples)

            for _ in range(ppo_epochs):
                permutation = torch.randperm(num_samples)
                for start_idx in range(0, num_samples, minibatch_size):
                    batch_idx = permutation[start_idx:start_idx + minibatch_size]
                    mb_obs = obs_batch[batch_idx]
                    mb_actions = action_batch[batch_idx]
                    mb_old_log_probs = old_log_prob_batch[batch_idx]
                    mb_advantages = advantage_batch[batch_idx]
                    mb_value_targets = value_target_batch[batch_idx]

                    logits, mb_values = model(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    mb_log_probs = dist.log_prob(mb_actions)
                    mb_entropy = dist.entropy().mean()

                    ratio = torch.exp(mb_log_probs - mb_old_log_probs)
                    unclipped = ratio * mb_advantages
                    clipped = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * mb_advantages
                    ppo_actor_loss = -torch.min(unclipped, clipped).mean()
                    ppo_critic_loss = F.mse_loss(mb_values, mb_value_targets)
                    ppo_loss = ppo_actor_loss + critic_weight * ppo_critic_loss - entropy_weight * mb_entropy

                    optimizer.zero_grad()
                    ppo_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            batch_obs.clear()
            batch_actions.clear()
            batch_old_log_probs.clear()
            batch_advantages.clear()
            batch_value_targets.clear()
            episodes_since_update = 0

        result.actor_loss = float(actor_loss.item())
        result.critic_loss = float(critic_loss.item())
        result.entropy = float(entropy_bonus.item())

        reward_history.append(result.reward)
        area_history.append(result.area)
        success_history.append(int(result.success))
        successful_area_history.append(result.area if result.success else 0.0)
        actor_loss_history.append(result.actor_loss)
        critic_loss_history.append(result.critic_loss)
        corner1_history.append(int(result.found_corner1))
        corner2_history.append(int(result.found_corner2))
        phase2_history.append(int(result.final_phase >= 2))
        straightness_history.append(result.triangle_straightness)
        extra_corner_history.append(result.extra_corners)
        good_triangle_history.append(int(result.good_triangle))
        shaping_reward_history.append(result.shaping_reward)
        terminal_reward_history.append(result.terminal_reward)
        terminal_area_bonus_history.append(result.terminal_area_bonus)
        terminal_good_bonus_history.append(result.terminal_good_bonus)
        stage_history.append(stage)
        training_results.append(result)

        if ep % 50 == 0 or ep == 1:
            avg_reward = float(np.mean(reward_history[-50:]))
            avg_area = float(np.mean(area_history[-50:]))
            success_rate = float(np.mean(success_history[-50:]) * 100.0)
            avg_actor_loss = float(np.mean(actor_loss_history[-50:]))
            avg_critic_loss = float(np.mean(critic_loss_history[-50:]))
            corner1_rate = float(np.mean(corner1_history[-50:]) * 100.0)
            corner2_rate = float(np.mean(corner2_history[-50:]) * 100.0)
            phase2_rate = float(np.mean(phase2_history[-50:]) * 100.0)
            mean_straightness = float(np.mean(straightness_history[-50:]))
            mean_extra_corners = float(np.mean(extra_corner_history[-50:]))
            mean_shaping_reward = float(np.mean(shaping_reward_history[-50:]))
            mean_terminal_reward = float(np.mean(terminal_reward_history[-50:]))
            mean_terminal_area_bonus = float(np.mean(terminal_area_bonus_history[-50:]))
            mean_terminal_good_bonus = float(np.mean(terminal_good_bonus_history[-50:]))
            eval_stats = evaluate_deterministic_policy(
                env=eval_env,
                model=model,
                gamma=gamma,
                gae_lambda=gae_lambda,
                episodes=periodic_eval_episodes,
            )
            print(
                f"Episode {ep:>3d} | "
                f"Stage={stage} | "
                f"Reward={result.reward:>8.3f} | "
                f"Area={result.area:>6.3f} | "
                f"Success={str(result.success):>5s} | "
                f"Phase={result.final_phase} | "
                f"C1={str(result.found_corner1):>5s} | "
                f"C2={str(result.found_corner2):>5s} | "
                f"Straight={result.triangle_straightness:>4.2f} | "
                f"Extra={result.extra_corners:>2d} | "
                f"Good={str(result.good_triangle):>5s} | "
                f"dStart={result.distance_to_start:>5.2f} | "
                f"ActorLoss={result.actor_loss:>8.3f} | "
                f"CriticLoss={result.critic_loss:>8.3f} | "
                f"letzte50: meanReward={avg_reward:>8.3f}, meanArea={avg_area:>6.3f}, "
                f"success={success_rate:>5.1f}%, c1={corner1_rate:>5.1f}%, c2={corner2_rate:>5.1f}%, "
                f"phase2={phase2_rate:>5.1f}%, evalSuccess={eval_stats['success_rate']:>5.1f}%, "
                f"evalArea={eval_stats['mean_area']:>6.3f}, meanActorLoss={avg_actor_loss:>8.3f}, meanCriticLoss={avg_critic_loss:>8.3f}"
                f", straight={mean_straightness:>4.2f}, extraCorners={mean_extra_corners:>4.2f}"
                f", shapeR={mean_shaping_reward:>7.2f}, termR={mean_terminal_reward:>7.2f}, areaB={mean_terminal_area_bonus:>7.2f}, goodB={mean_terminal_good_bonus:>7.2f}"
            )

        if ep >= 800 and (ep % checkpoint_eval_every == 0 or ep == episodes):
            checkpoint_env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
            checkpoint_env.set_curriculum_stage(3)
            checkpoint_results = collect_evaluation_rollouts(
                env=checkpoint_env,
                model=model,
                gamma=gamma,
                gae_lambda=gae_lambda,
                episodes=checkpoint_eval_episodes,
                sampling_temperature=0.75,
            )
            checkpoint_score = checkpoint_score_tuple(checkpoint_results)
            if checkpoint_score > best_checkpoint_score:
                best_checkpoint_score = checkpoint_score
                best_checkpoint_episode = ep
                best_checkpoint_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                print(
                    f"  Neuer bester Checkpoint @ Episode {ep}: "
                    f"clean={100.0 * checkpoint_score[0]:4.1f}%, "
                    f"good={100.0 * checkpoint_score[1]:4.1f}%, "
                    f"meanCleanArea={checkpoint_score[2]:.3f}, "
                    f"succ={100.0 * checkpoint_score[3]:4.1f}%, "
                    f"bestSuccArea={checkpoint_score[4]:.3f}"
                , flush=True)

    if best_checkpoint_state is not None:
        torch.save(
            {
                "episode": best_checkpoint_episode,
                "score": best_checkpoint_score,
                "state_dict": best_checkpoint_state,
            },
            best_checkpoint_path,
        )
        model.load_state_dict(best_checkpoint_state)
        print(
            f"\nLade besten Checkpoint aus Episode {best_checkpoint_episode} "
            f"fuer die Final-Evaluation: "
            f"clean={100.0 * best_checkpoint_score[0]:4.1f}%, "
            f"good={100.0 * best_checkpoint_score[1]:4.1f}%, "
            f"meanCleanArea={best_checkpoint_score[2]:.3f}, "
            f"succ={100.0 * best_checkpoint_score[3]:4.1f}%, "
            f"bestSuccArea={best_checkpoint_score[4]:.3f}"
        , flush=True)

    try:
        final_eval_env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
        final_eval_env.set_curriculum_stage(3)
        final_eval_results, final_eval_temperature_summaries = collect_temperature_sweep_rollouts(
            env=final_eval_env,
            model=model,
            gamma=gamma,
            gae_lambda=gae_lambda,
            temperatures=final_eval_temperatures,
            episodes_per_temperature=final_eval_episodes_per_temperature,
        )

        final_successes = [result for result in final_eval_results if result.success]
        final_good_successes = [result for result in final_successes if result.good_triangle]
        best_eval_success = max(final_successes, key=episode_quality_score, default=None)
        best_eval_overall = max(final_eval_results, key=episode_quality_score, default=None)

        summary_lines = [
            (
                f"Final Stage Evaluation ({len(final_eval_results)} stochastic rollouts "
                f"across T={', '.join(f'{temp:.2f}' for temp in final_eval_temperatures)})"
            )
        ]
        if final_successes:
            summary_lines.append(
                f"  success_rate={100.0 * len(final_successes) / len(final_eval_results):5.1f}% | "
                f"best_success_area={best_eval_success.area:.3f} | "
                f"best_success_straight={best_eval_success.triangle_straightness:.2f}"
            )
        else:
            summary_lines.append("  success_rate=  0.0% | no successful final-stage rollout found")

        if final_good_successes:
            best_good_success = max(final_good_successes, key=episode_quality_score)
            summary_lines.append(
                f"  good_triangle_rate={100.0 * len(final_good_successes) / len(final_eval_results):5.1f}% | "
                f"best_good_area={best_good_success.area:.3f} | "
                f"best_good_straight={best_good_success.triangle_straightness:.2f}"
            )
        else:
            summary_lines.append("  good_triangle_rate=  0.0% | no good final-stage triangle found")

        if best_eval_overall is not None:
            summary_lines.append(
                f"  best_overall: area={best_eval_overall.area:.3f}, "
                f"straight={best_eval_overall.triangle_straightness:.2f}, "
                f"extra_corners={best_eval_overall.extra_corners}, success={best_eval_overall.success}"
            )

        summary_lines.append("  Temperature sweep:")
        for summary in final_eval_temperature_summaries:
            summary_lines.append(
                f"    T={summary['temperature']:.2f} | "
                f"success={summary['success_rate']:5.1f}% | "
                f"good={summary['good_triangle_rate']:5.1f}% | "
                f"best_area={summary['best_area']:.3f} | "
                f"best_straight={summary['best_straightness']:.2f}"
            )

        print("\n" + "\n".join(summary_lines), flush=True)
        with open(final_summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write("\n".join(summary_lines) + "\n")

        plot_results(
            reward_history=reward_history,
            area_history=area_history,
            success_history=success_history,
            successful_area_history=successful_area_history,
            straightness_history=straightness_history,
            extra_corner_history=extra_corner_history,
            good_triangle_history=good_triangle_history,
            actor_loss_history=actor_loss_history,
            critic_loss_history=critic_loss_history,
            shaping_reward_history=shaping_reward_history,
            terminal_reward_history=terminal_reward_history,
            stage_history=stage_history,
            showcases=build_sampler_showcases(final_eval_results),
            progress_showcases=build_progress_showcases(training_results),
        )
    except Exception:
        print("\nFinal evaluation failed:\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
