#Unterstützung von KI in Design-Entscheidungen und teilweise bei Codeteilen

import random
import traceback
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent_helpers import (
    choose_antagonist_action,
    compute_gae,
    episode_area,
)
from fingerTriangleEnv import FingerTriangleEnv
from plotting_helpers import (
    build_progress_showcases,
    build_sampler_showcases,
    episode_quality_score,
    plot_results,
)

class ActorCriticNet(nn.Module):
    # Actor-Critic: gemeinsames Netz mit 2 Köpfen
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
    #Definition eines Episodenresults
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
    best_form_snapshot: dict | None = None
           
def run_episode(env: FingerTriangleEnv,model: ActorCriticNet,gamma: float,gae_lambda: float,antagonist_prob: float,deterministic: bool = False,sampling_temperature: float = 1.0,):
    #Eine Episode ausführen
    
    #Environment zurücksetzten
    obs, _ = env.reset()

    #Variablen initialisieren
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
    }

    while not (done or truncated):
        #Steps durchführen bis die Episode beendet wird
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = model(obs_t)
        
        #Auswahl der Aktion deterministisch oder nicht
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

        #Step durchführen
        next_obs, reward, done, truncated, info = env.step(action)

        #Werte aus Step speichern
        observations.append(np.array(obs, dtype=np.float32))
        actions.append(int(protagonist_action.item()))
        #logarithmus der Wahrscheinlichkeit der Aktion
        log_probs.append(dist.log_prob(protagonist_action).squeeze())
        #Bewertung des Critics des Zustands
        values.append(value.squeeze())
        #Streuung der Policy
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
        best_form_snapshot=env.bestFormSnapshot,
    )

    #Umwandlung in Tensoren für das Netz
    observations_t = torch.tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.int64)
    log_probs_t = torch.stack(log_probs).detach()
    entropies_t = torch.stack(entropies)
    advantages_t, value_targets_t = compute_gae(rewards=rewards, values=values, gamma=gamma, gae_lambda=gae_lambda)
    values_t = torch.stack(values)

    return result, observations_t, actions_t, log_probs_t, values_t, advantages_t, value_targets_t, entropies_t


def evaluate_deterministic_policy(
    env: FingerTriangleEnv,
    model: ActorCriticNet,
    gamma: float,
    gae_lambda: float,
    episodes: int = 20,
    deterministic: bool = True,
    sampling_temperature: float = 1.0,
    antagonist_prob: float = 0.0,
) -> dict[str, float]:
    #Bewertet den Outcome der Policy, standardmäßig deterministisch.
    rewards = []
    areas = []
    successes = []
    distances_to_start = []
    straightness_values = []

    with torch.no_grad():
        for _ in range(episodes):
            #Durchführung von Episoden mit deterministischer Policy
            result, _, _, _, _, _, _, _ = run_episode(
                env=env,
                model=model,
                gamma=gamma,
                gae_lambda=gae_lambda,
                antagonist_prob=antagonist_prob,
                deterministic=deterministic,
                sampling_temperature=sampling_temperature,
            )
            rewards.append(result.reward)
            areas.append(result.area)
            successes.append(int(result.success))
            distances_to_start.append(result.distance_to_start)
            straightness_values.append(result.triangle_straightness)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_area": float(np.mean(areas)),
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_distance_to_start": float(np.mean(distances_to_start)),
        "mean_straightness": float(np.mean(straightness_values)),
    }


def collect_evaluation_rollouts(env: FingerTriangleEnv, model: ActorCriticNet, gamma: float, gae_lambda: float, episodes: int = 60, sampling_temperature: float = 1.0) -> list[EpisodeResult]:
    #Ausführung von Episoden nach dem Training um die fertige Policy zu testen
    
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


def collect_temperature_sweep_rollouts(env: FingerTriangleEnv, model: ActorCriticNet, gamma: float, gae_lambda: float, temperatures: list[float], episodes_per_temperature: int) -> tuple[list[EpisodeResult], list[dict[str, float]]]:
    #Führt die Policy mit verschiedenen temperature-Werten aus um zu sehen, bei welcher Stärke von "greedyness"  die Policy am besten funktioniert
    
    all_results: list[EpisodeResult] = []
    summaries: list[dict[str, float]] = []

    next_episode = 1
    for temperature in temperatures:
        temp_results = collect_evaluation_rollouts(env=env, model=model, gamma=gamma, gae_lambda=gae_lambda, episodes=episodes_per_temperature, sampling_temperature=temperature)
        
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


def checkpoint_score_tuple(results: list[EpisodeResult]) -> tuple[float, float, float, float]:
    #Berechnet Werte, welche darüber entscheiden, ob das aktuelle Modell der bisher beste Checkpoint ist
    if not results:
        return (0.0, 0.0, 0.0, 0.0)

    successes = [result for result in results if result.success]
    good_successes = [result for result in successes if result.good_triangle]
    success_rate = len(successes) / len(results)
    good_rate = len(good_successes) / len(results)
    mean_success_area = float(np.mean([result.area for result in successes])) if successes else 0.0
    best_success_area = max((result.area for result in successes), default=0.0)
    return (good_rate, mean_success_area, success_rate, best_success_area)


# -----------------------------
# Hauptlauf
# -----------------------------
def main():
    # Reproduzierbarkeit
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #Variablen um Laufdauer zu entscheiden
    quick_test = os.environ.get("FT_QUICK_TEST", "1") == "1"
    long_run = os.environ.get("FT_LONG_RUN", "0") == "1"
    fast_tune = os.environ.get("FT_FAST_TUNE", "0") == "1"
    disable_plot = os.environ.get("FT_DISABLE_PLOT", "0") == "1"

    # Hyperparameter
    if quick_test:
        episodes = 1200
    elif long_run:
        episodes = 6000
    else:
        episodes = 3000

    if fast_tune:
        episodes = int(os.environ.get("FT_EPISODES", "450"))
    
    #einstellbare Parameter für Lernveränderung    
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    critic_weight = 0.5
    entropy_weight_start = 0.0012
    entropy_weight_end = 0.00005
    update_batch_episodes = 8
    ppo_epochs = 4
    ppo_minibatch_size = 256
    ppo_clip_epsilon = 0.2
    antagonist_prob = 0.08
    use_antagonist = True
    final_eval_temperatures = [0.75]
    final_eval_episodes_per_temperature = 80 if quick_test else 360
    checkpoint_eval_every = 200 if quick_test else 100
    checkpoint_eval_episodes = 16 if quick_test else 48
    periodic_eval_episodes = 5 if quick_test else 15

    if fast_tune:
        final_eval_episodes_per_temperature = int(os.environ.get("FT_FINAL_EVAL_EPISODES", "40"))
        checkpoint_eval_every = int(os.environ.get("FT_CHECKPOINT_EVERY", "100"))
        checkpoint_eval_episodes = int(os.environ.get("FT_CHECKPOINT_EPISODES", "12"))
        periodic_eval_episodes = int(os.environ.get("FT_PERIODIC_EVAL_EPISODES", "4"))
    best_checkpoint_path = "best_triangle_checkpoint.pt"
    final_summary_path = "final_eval_summary.txt"

    #Initialisierung von Environments
    env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
    eval_env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
    benchmark_env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
    benchmark_env.set_curriculum_stage(3)
    obs_dim = int(np.prod(env.observation_space.shape))

    #Initialisierung des Netzes
    model = ActorCriticNet(obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Initialisierung von Speicherlisten
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
    stage_history: list[int] = []
    training_results: list[EpisodeResult] = []
    benchmark_episode_points: list[int] = []
    benchmark_reward_history: list[float] = []
    benchmark_success_history: list[float] = []
    benchmark_area_history: list[float] = []
    benchmark_distance_history: list[float] = []
    benchmark_straightness_history: list[float] = []
    benchmark_stage_history: list[int] = []
    best_checkpoint_state: dict[str, torch.Tensor] | None = None
    best_checkpoint_score = (-1.0, -1.0, -1.0, -1.0)
    best_checkpoint_episode = 0

    #Stage-definition für verschiedene Run-Optionen
    if fast_tune:
        stage_endpoints = (
            int(0.27 * episodes),
            int(0.60 * episodes),
            int(0.90 * episodes),
            episodes,
        )
    elif quick_test:
        stage_endpoints = (320, 720, 1080, episodes)
    elif long_run:
        stage_endpoints = (1600, 3800, 5600, episodes)
    else:
        stage_endpoints = (800, 1900, 2800, episodes)
       
    #Einteilung von stage3 in zwei Phasen 
    stage3_refinement_start = stage_endpoints[2] + int(0.45 * (episodes - stage_endpoints[2]))
    stage3_final_start = stage_endpoints[2] + int(0.70 * (episodes - stage_endpoints[2]))
    stage3_bridge_end = stage_endpoints[2] + int(0.32 * (episodes - stage_endpoints[2]))

    def apply_stage3_bridge(curr_env: FingerTriangleEnv) -> None:
        # Zu Beginn von Stage 3 wird die Verschärfung etwas abgefedert,
        # damit gute Stage-2-Politiken nicht sofort einbrechen.
        curr_env.closureRadius = 1.65
        curr_env.minStraightnessForSuccess = 0.38
        curr_env.minMeanEdgeLengthForGoodTriangle = 0.95
        curr_env.minEdgeBalanceForGoodTriangle = 0.40
        curr_env.reward_Phase2DirectionAlignment = 2.0
        curr_env.reward_Phase2CleanReturn = 1.0
        curr_env.penalty_Phase2AwayFromStart = 7.5
        curr_env.penalty_Phase2ReturnLineDeviation = 5.0
        curr_env.penalty_Phase2LateDistance = 1.2
        curr_env.partialAreaScale = 0.40
        curr_env.terminalGapScale = 1.00

    def curriculum_stage_for_episode(episode: int) -> int:
        #Gibt Lernstage für übergebene Episode zurück
        if episode <= stage_endpoints[0]:
            return 0
        if episode <= stage_endpoints[1]:
            return 1
        if episode <= stage_endpoints[2]:
            return 2
        return 3

    print("Starte Actor-Critic-Lauf ...\n")

    if quick_test:
        print("Modus: 'Quick Test' \n")
    elif long_run:
        print("Modus: 'Long Run' \n")
    else:
        print("Modus: 'Normal Run' \n")

    #Werte-Sammlungen initialisieren
    episodes_since_update = 0
    batch_obs: list[torch.Tensor] = []
    batch_actions: list[torch.Tensor] = []
    batch_old_log_probs: list[torch.Tensor] = []
    batch_advantages: list[torch.Tensor] = []
    batch_value_targets: list[torch.Tensor] = []

    for ep in range(1, episodes + 1):
        #Durchführung der angegebenen Anzahl an Episoden
        
        #Lern-Stage setzen
        stage = curriculum_stage_for_episode(ep)
        env.set_curriculum_stage(stage)
        eval_env.set_curriculum_stage(stage)
        if stage == 3 and ep < stage3_bridge_end:
            apply_stage3_bridge(env)
            apply_stage3_bridge(eval_env)

        #Durchführung der Episode
        result, observations, actions, old_log_probs, values, advantages, value_targets, entropies = run_episode(env, model, gamma, gae_lambda, antagonist_prob)
        result.episode = ep
        result.curriculum_stage = stage

        #Actor und critic losses berechnen
        logits, predicted_values = model(observations)
        dist = torch.distributions.Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions)
        ratio = torch.exp(current_log_probs - old_log_probs)
        unclipped = ratio * advantages.detach()
        clipped = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages.detach()
        actor_loss = -torch.min(unclipped, clipped).mean()
        critic_loss = F.mse_loss(predicted_values, value_targets)
        
        #training-Parameter aktualisieren
        entropy_bonus = entropies.mean()
        progress = (ep - 1) / max(1, episodes - 1)
        entropy_weight = entropy_weight_start + progress * (entropy_weight_end - entropy_weight_start)
        current_lr = learning_rate
        
        #Learning rate am Ende runtersetzen um nur Feinjustierungen durchzuführen ohne die Policy instabil zu machen
        current_ppo_epochs = ppo_epochs
        if stage == 3:
            if ep < stage3_bridge_end:
                current_lr = learning_rate * 0.70
                entropy_weight *= 0.70
            elif ep >= stage3_final_start:
                current_lr = learning_rate * 0.18
                entropy_weight *= 0.15
                current_ppo_epochs = 1
            elif ep >= stage3_refinement_start:
                current_lr = learning_rate * 0.35
                entropy_weight *= 0.35
                current_ppo_epochs = max(2, ppo_epochs - 2)

        #Aktualisieren der Learning Rate
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        #Episoden-Metriken in sammlungen hinzufügen
        batch_obs.append(observations.detach())
        batch_actions.append(actions.detach())
        batch_old_log_probs.append(old_log_probs.detach())
        batch_advantages.append(advantages.detach())
        batch_value_targets.append(value_targets.detach())
        episodes_since_update += 1

        #Wenn bestimmte Episodenanzahl erreich ist wird das Netzwerk aktualisiert
        if episodes_since_update >= update_batch_episodes or ep == episodes:
            
            #Alle gesammelten Tensoren zu einem Tensor zusammenfügen
            obs_batch = torch.cat(batch_obs, dim=0)
            action_batch = torch.cat(batch_actions, dim=0)
            old_log_prob_batch = torch.cat(batch_old_log_probs, dim=0)
            advantage_batch = torch.cat(batch_advantages, dim=0)
            value_target_batch = torch.cat(batch_value_targets, dim=0)

            num_samples = obs_batch.shape[0]
            minibatch_size = min(ppo_minibatch_size, num_samples)

            #Optimierung des PPo-Netzes
            for _ in range(current_ppo_epochs):
                permutation = torch.randperm(num_samples)
                #Aufteilen der Batches in kleine Sammlungen
                for start_idx in range(0, num_samples, minibatch_size):
                    #kleine Sammlung wählen
                    batch_idx = permutation[start_idx:start_idx + minibatch_size]
                    
                    #kleine Batches aus den großen Batches nehmen
                    mb_obs = obs_batch[batch_idx]
                    mb_actions = action_batch[batch_idx]
                    mb_old_log_probs = old_log_prob_batch[batch_idx]
                    mb_advantages = advantage_batch[batch_idx]
                    mb_value_targets = value_target_batch[batch_idx]

                    #Netzwerk auf den kleinen Sammlungen laufen lassen
                    logits, mb_values = model(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    mb_log_probs = dist.log_prob(mb_actions)
                    mb_entropy = dist.entropy().mean()

                    #Losses berechnen
                    ratio = torch.exp(mb_log_probs - mb_old_log_probs)
                    unclipped = ratio * mb_advantages
                    clipped = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * mb_advantages
                    ppo_actor_loss = -torch.min(unclipped, clipped).mean()
                    ppo_critic_loss = F.mse_loss(mb_values, mb_value_targets)
                    ppo_loss = ppo_actor_loss + critic_weight * ppo_critic_loss - entropy_weight * mb_entropy

                    #Aktualisieren der Modelparameter
                    optimizer.zero_grad()
                    ppo_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            #Tensoreninhalte löschen
            batch_obs.clear()
            batch_actions.clear()
            batch_old_log_probs.clear()
            batch_advantages.clear()
            batch_value_targets.clear()
            episodes_since_update = 0

        result.actor_loss = float(actor_loss.item())
        result.critic_loss = float(critic_loss.item())
        result.entropy = float(entropy_bonus.item())

        #Metriken in Speicher anhängen
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
        stage_history.append(stage)
        training_results.append(result)

        #BErechnung Ausgabe von Benchmarks alle 50 Episoden
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
            eval_stats = evaluate_deterministic_policy(
                env=eval_env,
                model=model,
                gamma=gamma,
                gae_lambda=gae_lambda,
                episodes=periodic_eval_episodes,
                deterministic=True,
            )
            stage3_benchmark_stats = evaluate_deterministic_policy(
                env=benchmark_env,
                model=model,
                gamma=gamma,
                gae_lambda=gae_lambda,
                episodes=max(periodic_eval_episodes, 24),
                deterministic=False,
                sampling_temperature=0.75,
                antagonist_prob=0.0,
            )
            
            #Benchmark-Werte in Speicher anhängen
            benchmark_episode_points.append(ep)
            benchmark_reward_history.append(stage3_benchmark_stats["mean_reward"])
            benchmark_success_history.append(stage3_benchmark_stats["success_rate"])
            benchmark_area_history.append(stage3_benchmark_stats["mean_area"])
            benchmark_distance_history.append(stage3_benchmark_stats["mean_distance_to_start"])
            benchmark_straightness_history.append(stage3_benchmark_stats["mean_straightness"])
            benchmark_stage_history.append(stage)
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
                f"evalArea={eval_stats['mean_area']:>6.3f}, stage3BenchSucc={stage3_benchmark_stats['success_rate']:>5.1f}%, "
                f"stage3BenchR={stage3_benchmark_stats['mean_reward']:>7.1f}, "
                f"stage3BenchD={stage3_benchmark_stats['mean_distance_to_start']:>5.2f}, "
                f"stage3BenchStr={stage3_benchmark_stats['mean_straightness']:>4.2f}, "
                f"meanActorLoss={avg_actor_loss:>8.3f}, meanCriticLoss={avg_critic_loss:>8.3f}"
                f", straight={mean_straightness:>4.2f}, extraCorners={mean_extra_corners:>4.2f}"
                f", shapeR={mean_shaping_reward:>7.2f}, termR={mean_terminal_reward:>7.2f}, areaB={mean_terminal_area_bonus:>7.2f}"
            )

        #nach Episode 800 wird regelmäßig geprüft ob die Episode der beste neue Checkpoint ist
        if ep >= 800 and (ep % checkpoint_eval_every == 0 or ep == episodes):
            checkpoint_env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
            checkpoint_env.set_curriculum_stage(3)
            checkpoint_results = collect_evaluation_rollouts(env=checkpoint_env, model=model, gamma=gamma, gae_lambda=gae_lambda, episodes=checkpoint_eval_episodes, sampling_temperature=0.75)
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
                    f"good={100.0 * checkpoint_score[0]:4.1f}%, "
                    f"meanSuccArea={checkpoint_score[1]:.3f}, "
                    f"succ={100.0 * checkpoint_score[2]:4.1f}%, "
                    f"bestSuccArea={checkpoint_score[3]:.3f}"
                , flush=True)

    #Nach Episoden den besten Checkpoint laden
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
            f"good={100.0 * best_checkpoint_score[0]:4.1f}%, "
            f"meanSuccArea={best_checkpoint_score[1]:.3f}, "
            f"succ={100.0 * best_checkpoint_score[2]:4.1f}%, "
            f"bestSuccArea={best_checkpoint_score[3]:.3f}"
        , flush=True)

    try:
        #mit gelernter Policy noch einige Episoden durchführen und beste Formen speichern
        final_eval_env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
        final_eval_env.set_curriculum_stage(3)
        final_eval_results, final_eval_temperature_summaries = collect_temperature_sweep_rollouts(env=final_eval_env, model=model, gamma=gamma, gae_lambda=gae_lambda, temperatures=final_eval_temperatures, episodes_per_temperature=final_eval_episodes_per_temperature)

        final_successes = [result for result in final_eval_results if result.success]
        final_good_successes = [result for result in final_successes if result.good_triangle]
        best_eval_success = max(final_successes, key=episode_quality_score, default=None)
        best_eval_overall = max(final_eval_results, key=episode_quality_score, default=None)

        #Output der benchmarks der letzten Rollouts
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

        #Plotting
        if not disable_plot:
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
                distance_history=[result.distance_to_start for result in training_results],
                training_results=training_results,
                benchmark_episode_points=benchmark_episode_points,
                benchmark_reward_history=benchmark_reward_history,
                benchmark_success_history=benchmark_success_history,
                benchmark_area_history=benchmark_area_history,
                benchmark_distance_history=benchmark_distance_history,
                benchmark_straightness_history=benchmark_straightness_history,
                benchmark_stage_history=benchmark_stage_history,
                showcases=build_sampler_showcases(final_eval_results),
                progress_showcases=build_progress_showcases(training_results),
            )
    except Exception:
        print("\nFinal evaluation failed:\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
