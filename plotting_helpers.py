#Plotting wurde mit KI unterstützt

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from agent_helpers import (
    moving_average,
    moving_success_average,
    stagewise_benchmark_curves,
    stagewise_metric_curves,
)

if TYPE_CHECKING:
    from actor_critic_finger import EpisodeResult


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


def visually_closed(result: EpisodeResult) -> bool:
    return bool(result.success and result.distance_to_start <= 0.60)


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
    stage_labels = {0: "Beste Form Stage 0", 1: "Beste Form Stage 1", 2: "Beste Form Stage 2", 3: "Beste Form Stage 3"}
    for stage in [0, 1, 2, 3]:
        stage_results = [result for result in results if result.curriculum_stage == stage]
        if not stage_results:
            continue
        stage_successes = [result for result in stage_results if result.success]
        source = stage_successes if stage_successes else stage_results
        candidate = best_form_variant(max(source, key=best_form_quality_score))
        selected.append((stage_labels[stage], candidate))
    return build_showcase_episodes(selected, limit=4)


def build_recent_triangle_showcases(results: list[EpisodeResult], limit: int = 3) -> list[tuple[str, EpisodeResult]]:
    if not results:
        return []
    triangle_like = [result for result in results if result.found_corner2 or result.final_phase >= 2]
    source = triangle_like if len(triangle_like) >= limit else results
    selected = source[-limit:]
    labels = [f"Letzte {idx + 1}" for idx in range(len(selected))]
    return list(zip(labels, selected))


def summarize_training_metrics(
    reward_history: list[float],
    area_history: list[float],
    success_history: list[int],
    good_triangle_history: list[int],
    distance_history: list[float],
    successful_area_history: list[float],
    stage_history: list[int],
    benchmark_reward_history: list[float],
    benchmark_success_history: list[float],
    benchmark_area_history: list[float],
) -> list[str]:
    if not reward_history:
        return ["Keine Trainingsdaten vorhanden."]

    latest_window = min(50, len(reward_history))
    reward_ma = moving_average(reward_history, 25)
    success_ma = moving_average(success_history, 25) * 100.0
    good_ma = moving_average(good_triangle_history, 25) * 100.0
    success_area_ma = moving_success_average(successful_area_history, success_history, 25)
    closure_gap_ma = moving_average(distance_history, 25)

    stage_lines: list[str] = []
    stages = np.asarray(stage_history, dtype=np.int32)
    rewards = np.asarray(reward_history, dtype=np.float32)
    successes = np.asarray(success_history, dtype=np.float32)
    good = np.asarray(good_triangle_history, dtype=np.float32)
    areas = np.asarray(area_history, dtype=np.float32)

    for stage in sorted(set(int(s) for s in stages.tolist())):
        idx = np.where(stages == stage)[0]
        if len(idx) == 0:
            continue
        stage_lines.append(
            f"Stage {stage} lokal: Reward {np.mean(rewards[idx]):.1f}, "
            f"Success {100.0 * np.mean(successes[idx]):.1f}%, "
            f"Good {100.0 * np.mean(good[idx]):.1f}%, "
            f"Area {np.mean(areas[idx]):.3f}"
        )

    benchmark_lines: list[str] = []
    if benchmark_reward_history:
        benchmark_lines.extend([
            f"Stage-3 Benchmark R: {benchmark_reward_history[-1]:.1f}",
            f"Stage-3 Benchmark S: {benchmark_success_history[-1]:.1f}%",
            f"Stage-3 Benchmark A: {benchmark_area_history[-1]:.3f}",
        ])

    return [
        "Metrik-Hinweis:",
        "Lokal = nach aktuellem Stage-Standard",
        "Benchmark = immer auf Stage 3 bewertet",
        f"Letzte {latest_window} Ep. lokal R: {np.mean(reward_history[-latest_window:]):.1f}",
        f"Letzte {latest_window} Ep. lokal S: {100.0 * np.mean(success_history[-latest_window:]):.1f}%",
        f"Letzte {latest_window} Ep. lokal G: {100.0 * np.mean(good_triangle_history[-latest_window:]):.1f}%",
        f"MA25 lokal Reward: {reward_ma[-1]:.1f}",
        f"MA25 lokal Success: {success_ma[-1]:.1f}%",
        f"MA25 lokal Good: {good_ma[-1]:.1f}%",
        f"MA25 Erfolgsflaeche: {success_area_ma[-1]:.3f}",
        f"MA25 Abschlussluecke: {closure_gap_ma[-1]:.3f}",
        f"Beste Flaeche bisher: {max(area_history):.3f}",
        f"Bester Erfolg bisher: {max(successful_area_history):.3f}",
        *benchmark_lines,
        *stage_lines,
    ]


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
    stage_colors = {0: "#eaf4d3", 1: "#d8f3dc", 2: "#dceaf7", 3: "#fde2e4"}
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
    distance_history,
    training_results,
    benchmark_episode_points,
    benchmark_reward_history,
    benchmark_success_history,
    benchmark_area_history,
    benchmark_stage_history,
    showcases,
    progress_showcases,
):
    reward_ma = moving_average(reward_history, 25)
    success_ma = moving_average(success_history, 25) * 100.0
    successful_area_ma = moving_success_average(successful_area_history, success_history, 25)
    straightness_ma = moving_average(straightness_history, 25)
    extra_corner_ma = moving_average(extra_corner_history, 25)
    actor_loss_ma = moving_average(actor_loss_history, 25)
    critic_loss_ma = moving_average(critic_loss_history, 25)
    shaping_reward_ma = moving_average(shaping_reward_history, 25)
    terminal_reward_ma = moving_average(terminal_reward_history, 25)
    closure_gap_ma = moving_average(distance_history, 25)
    stage_reward_curves = stagewise_metric_curves(reward_history, stage_history, 25)
    stage_success_curves = stagewise_metric_curves(success_history, stage_history, 25, multiplier=100.0)
    stage_area_curves = stagewise_metric_curves(area_history, stage_history, 25)
    benchmark_success_curves = stagewise_benchmark_curves(benchmark_episode_points, benchmark_success_history, benchmark_stage_history, window=3, multiplier=1.0)
    benchmark_reward_ma = moving_average(benchmark_reward_history, 3) if benchmark_reward_history else np.array([])
    benchmark_area_ma = moving_average(benchmark_area_history, 3) if benchmark_area_history else np.array([])
    recent_showcases = build_recent_triangle_showcases(training_results, limit=3)
    summary_lines = summarize_training_metrics(
        reward_history=reward_history,
        area_history=area_history,
        success_history=success_history,
        good_triangle_history=good_triangle_history,
        distance_history=distance_history,
        successful_area_history=successful_area_history,
        stage_history=stage_history,
        benchmark_reward_history=benchmark_reward_history,
        benchmark_success_history=benchmark_success_history,
        benchmark_area_history=benchmark_area_history,
    )

    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 9), num="Training Trends")
    fig1.suptitle("Training Trends", fontsize=16)

    ax1 = axs1[0, 0]
    ax1.plot(reward_history, alpha=0.25, label="Reward pro Episode")
    ax1.plot(reward_ma, linewidth=2.5, label="Moving Average (25)")
    add_stage_background(ax1, stage_history)
    ax1.set_title("Reward über die Zeit")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axs1[0, 1]
    stage_colors = {0: "tab:green", 1: "tab:blue", 2: "tab:orange", 3: "tab:red"}
    for stage, (x_vals, y_vals) in stage_reward_curves.items():
        ax2.plot(x_vals, y_vals, linewidth=2.2, color=stage_colors.get(stage, None), label=f"Stage {stage}")
    ax2.set_title("Reward je Stage (MA 25)")
    ax2.set_xlabel("Episode innerhalb der Stage")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axs1[1, 0]
    ax3.plot(success_ma, linewidth=2.5)
    add_stage_background(ax3, stage_history)
    ax3.set_title("Success-Rate lokal (aktueller Stage-Standard, MA 25)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Erfolgsquote in %")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    ax4 = axs1[1, 1]
    ax4.plot(closure_gap_ma, linewidth=2.5, color="tab:red", label="dStart MA")
    add_stage_background(ax4, stage_history)
    ax4.set_title("Abschlussluecke zum Start (MA 25)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("dStart")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 9), num="Benchmark And Quality")
    fig2.suptitle("Benchmark And Quality", fontsize=16)

    ax5 = axs2[0, 0]
    ax5.plot(successful_area_ma, linewidth=2.5, color="tab:orange", label="Erfolgsflaeche MA")
    ax5.plot(area_history, alpha=0.08, color="tab:orange")
    add_stage_background(ax5, stage_history)
    ax5.set_title("Mittlere Erfolgsflaeche lokal")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Area bei Erfolgen")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = axs2[0, 1]
    for stage, (x_vals, y_vals) in stage_success_curves.items():
        ax6.plot(x_vals, y_vals, linewidth=2.2, color=stage_colors.get(stage, None), label=f"Stage {stage}")
    ax6.set_title("Success je Stage lokal (MA 25)")
    ax6.set_xlabel("Episode innerhalb der Stage")
    ax6.set_ylabel("Erfolgsquote in %")
    ax6.set_ylim(0, 100)
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    ax7 = axs2[1, 0]
    if benchmark_episode_points:
        ax7.plot(benchmark_episode_points, benchmark_reward_history, alpha=0.20, color="tab:purple", label="Stage-3 Benchmark Reward")
        ax7.plot(benchmark_episode_points, benchmark_reward_ma, linewidth=2.5, color="tab:purple", label="Benchmark Reward MA")
    add_stage_background(ax7, stage_history)
    ax7.set_title("Reward auf fixem Stage-3-Standard")
    ax7.set_xlabel("Episode")
    ax7.set_ylabel("Benchmark Reward")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    ax8 = axs2[1, 1]
    ax8.plot(critic_loss_history, alpha=0.25, label="Critic Loss")
    ax8.plot(critic_loss_ma, linewidth=2.5, label="Critic Loss MA")
    ax8.plot(actor_loss_history, alpha=0.2, label="Actor Loss")
    ax8.plot(actor_loss_ma, linewidth=2.0, label="Actor Loss MA")
    ax8_twin = ax8.twinx()
    ax8_twin.plot(shaping_reward_ma, linewidth=1.6, color="tab:blue", alpha=0.8, label="Shape Reward MA")
    ax8_twin.plot(terminal_reward_ma, linewidth=1.6, color="tab:purple", alpha=0.8, label="Terminal Reward MA")
    add_stage_background(ax8, stage_history)
    ax8.set_title("Verluste und Reward-Anteile")
    ax8.set_xlabel("Episode")
    ax8.set_ylabel("Loss")
    ax8_twin.set_ylabel("Reward-Anteile")
    ax8.grid(True, alpha=0.3)
    lines_left, labels_left = ax8.get_legend_handles_labels()
    lines_right, labels_right = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="upper right")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    fig3, axs3 = plt.subplots(2, 2, figsize=(15, 9), num="Stage-3 Standard And Shape")
    fig3.suptitle("Stage-3 Standard And Shape", fontsize=16)

    ax9 = axs3[0, 0]
    ax9.plot(straightness_history, alpha=0.25, label="Straightness")
    ax9.plot(straightness_ma, linewidth=2.5, label="Straightness MA")
    ax9_twin = ax9.twinx()
    ax9_twin.plot(extra_corner_history, alpha=0.2, color="tab:red", label="Extra Corners")
    ax9_twin.plot(extra_corner_ma, linewidth=2.0, color="tab:red", label="Extra Corners MA")
    add_stage_background(ax9, stage_history)
    ax9.set_title("Formqualität")
    ax9.set_xlabel("Episode")
    ax9.set_ylabel("Straightness")
    ax9_twin.set_ylabel("Extra Corners")
    ax9.grid(True, alpha=0.3)
    lines_left, labels_left = ax9.get_legend_handles_labels()
    lines_right, labels_right = ax9_twin.get_legend_handles_labels()
    ax9.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="upper right")

    ax10 = axs3[0, 1]
    for stage, (x_vals, y_vals) in stage_area_curves.items():
        ax10.plot(x_vals, y_vals, linewidth=2.2, color=stage_colors.get(stage, None), label=f"Stage {stage}")
    ax10.set_title("Flaeche je Stage lokal (MA 25)")
    ax10.set_xlabel("Episode innerhalb der Stage")
    ax10.set_ylabel("Fläche")
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    ax11 = axs3[1, 0]
    for stage, (x_vals, y_vals) in benchmark_success_curves.items():
        ax11.plot(x_vals, y_vals, linewidth=2.2, color=stage_colors.get(stage, None), label=f"Train in Stage {stage}")
    ax11.set_title("Stage-3-Benchmark-Success je Trainings-Stage")
    ax11.set_xlabel("Episode")
    ax11.set_ylabel("Erfolgsquote in %")
    ax11.set_ylim(0, 100)
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    ax12 = axs3[1, 1]
    if benchmark_episode_points:
        ax12.plot(benchmark_episode_points, benchmark_area_history, alpha=0.20, color="tab:orange", label="Stage-3 Benchmark Area")
        ax12.plot(benchmark_episode_points, benchmark_area_ma, linewidth=2.5, color="tab:orange", label="Benchmark Area MA")
    add_stage_background(ax12, stage_history)
    ax12.set_title("Flaeche auf fixem Stage-3-Standard")
    ax12.set_xlabel("Episode")
    ax12.set_ylabel("Area")
    ax12.grid(True, alpha=0.3)
    ax12.legend()
    fig3.tight_layout(rect=[0, 0, 1, 0.96])

    fig4, axs4 = plt.subplots(1, 4, figsize=(18, 5), num="Best Per Stage")
    fig4.suptitle("Best Per Stage", fontsize=16)
    progress_handles = []
    progress_labels = []
    if progress_showcases:
        for idx, (label, result) in enumerate(progress_showcases[:4]):
            ax = axs4[idx]
            plot_episode(ax, label, result)
            if not progress_handles:
                progress_handles, progress_labels = ax.get_legend_handles_labels()
    else:
        for ax in axs4:
            ax.axis("off")
    if progress_handles:
        fig4.legend(progress_handles, progress_labels, loc="lower center", ncol=4)
    fig4.tight_layout(rect=[0, 0.05, 1, 0.94])

    fig5, axs5 = plt.subplots(1, 4, figsize=(18, 5), num="Best Final Triangles")
    fig5.suptitle("Best Final Triangles", fontsize=16)
    showcase_handles = []
    showcase_labels = []
    if showcases:
        for idx, (label, result) in enumerate(showcases[:4]):
            ax = axs5[idx]
            plot_episode(ax, label, result)
            if not showcase_handles:
                showcase_handles, showcase_labels = ax.get_legend_handles_labels()
    else:
        for ax in axs5:
            ax.axis("off")
        axs5[1].text(0.5, 0.5, "Keine erfolgreichen Final-Stage-Rollouts gefunden.", ha="center", va="center", fontsize=14)
    if showcase_handles:
        fig5.legend(showcase_handles, showcase_labels, loc="lower center", ncol=4)
    fig5.tight_layout(rect=[0, 0.05, 1, 0.94])

    fig6, axs6 = plt.subplots(1, 4, figsize=(18, 5), num="Recent Triangles And Summary")
    fig6.suptitle("Recent Triangles And Summary", fontsize=16)
    if recent_showcases:
        for idx, (label, result) in enumerate(recent_showcases[:3]):
            plot_episode(axs6[idx], label, result)
    for idx in range(len(recent_showcases), 3):
        axs6[idx].axis("off")

    summary_ax = axs6[3]
    summary_ax.axis("off")
    summary_ax.set_title("Kennzahlen")
    summary_ax.text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", fontsize=10, family="monospace")
    fig6.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
