import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fingerTriangleEnv import FingerTriangleEnv


# -----------------------------
# Kleine, intuitive Policy
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def choose_antagonist_action(prob: float = 0.12) -> int:
    """
    Einfacher bounded Antagonist:
    - meistens tut er nichts (Aktion 0)
    - manchmal stört er mit einer kleinen zufälligen 27er-Aktion
    """
    if random.random() < prob:
        return random.randint(0, 26)
    return 0


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def episode_area(env: FingerTriangleEnv) -> float:
    if env.corner1 is None or env.corner2 is None:
        return 0.0
    try:
        return float(env.calculateTriangleArea(env.startPos, env.corner1, env.corner2))
    except Exception:
        return 0.0


def run_episode(env: FingerTriangleEnv, policy: PolicyNet, gamma: float, antagonist_prob: float):
    obs, _ = env.reset()
    log_probs = []
    rewards = []

    done = False
    truncated = False

    while not (done or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        protagonist_action = dist.sample()

        action = {
            "protagonist": int(protagonist_action.item()),
            "antagonist": choose_antagonist_action(antagonist_prob),
        }

        obs, reward, done, truncated, _ = env.step(action)
        log_probs.append(dist.log_prob(protagonist_action))
        rewards.append(float(reward))

    result = EpisodeResult(
        episode=0,
        reward=float(sum(rewards)),
        area=episode_area(env),
        success=bool(done),
        positions=np.array(env.positionSaver, dtype=np.float32),
        start=np.array(env.startPos, dtype=np.float32),
        corner1=None if env.corner1 is None else np.array(env.corner1, dtype=np.float32),
        corner2=None if env.corner2 is None else np.array(env.corner2, dtype=np.float32),
    )

    returns = discounted_returns(rewards, gamma)
    return result, log_probs, returns


# -----------------------------
# Hauptlauf
# -----------------------------
def main():
    # Reproduzierbarkeit
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    episodes = 600
    gamma = 0.99
    learning_rate = 2e-3
    antagonist_prob = 0.10

    env = FingerTriangleEnv(givenUseAntagonist=True)
    obs_dim = int(np.prod(env.observation_space.shape))

    policy = PolicyNet(obs_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    reward_history: list[float] = []
    area_history: list[float] = []
    success_history: list[int] = []
    snapshots: dict[int, EpisodeResult] = {}

    snapshot_episodes = {1, 10, 50, 100, 200, 500, episodes}

    print("Starte Demo-Lauf ...")
    print("Es wird nichts gespeichert und kein separates Modell trainiert.")
    print("Die Policy lernt nur in diesem einen Skriptlauf und die Entwicklung wird danach geplottet.\n")

    baseline = 0.0
    baseline_momentum = 0.9

    for ep in range(1, episodes + 1):
        result, log_probs, returns = run_episode(env, policy, gamma, antagonist_prob)
        result.episode = ep

        # Einfaches REINFORCE mit gleitendem Baseline-Wert
        episode_return = result.reward
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * episode_return
        advantage_shift = baseline / max(1.0, abs(baseline))
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * (G - advantage_shift))
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        reward_history.append(result.reward)
        area_history.append(result.area)
        success_history.append(int(result.success))

        if ep in snapshot_episodes:
            snapshots[ep] = result

        if ep % 50 == 0 or ep == 1:
            avg_reward = np.mean(reward_history[-50:])
            avg_area = np.mean(area_history[-50:])
            success_rate = np.mean(success_history[-50:]) * 100.0
            print(
                f"Episode {ep:>3d} | "
                f"Reward={result.reward:>7.3f} | "
                f"Area={result.area:>6.3f} | "
                f"Success={str(result.success):>5s} | "
                f"letzte50: meanReward={avg_reward:>7.3f}, meanArea={avg_area:>6.3f}, success={success_rate:>5.1f}%"
            )

    plot_results(reward_history, area_history, success_history, snapshots)


# -----------------------------
# Visualisierung
# -----------------------------
def moving_average(x: list[float], window: int = 25) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    ma = np.convolve(x, kernel, mode="valid")
    pad = np.full(window - 1, ma[0], dtype=np.float32)
    return np.concatenate([pad, ma])



def plot_results(reward_history, area_history, success_history, snapshots):
    reward_ma = moving_average(reward_history, 25)
    area_ma = moving_average(area_history, 25)
    success_ma = moving_average(success_history, 25) * 100.0

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.3], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(reward_history, alpha=0.35, linewidth=1.0, label="Reward pro Episode")
    ax1.plot(reward_ma, linewidth=2.2, label="Moving Average (25)")
    ax1.set_title("Reward über die Zeit")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(area_history, alpha=0.35, linewidth=1.0, label="Fläche pro Episode")
    ax2.plot(area_ma, linewidth=2.2, label="Moving Average (25)")
    ax2.set_title("Dreiecksfläche über die Zeit")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Fläche")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(success_ma, linewidth=2.2)
    ax3.set_title("Success-Rate (Moving Average 25)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Erfolgsquote in %")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.25)

    ordered_eps = [ep for ep in sorted(snapshots.keys()) if ep in snapshots]
    n = len(ordered_eps)
    cols = min(4, max(1, n))
    rows = math.ceil(n / cols)
    subgs = gs[1, :].subgridspec(rows, cols, hspace=0.45, wspace=0.35)

    legend_handles = None
    legend_labels = None

    for i, ep in enumerate(ordered_eps):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(subgs[row, col])

        res = snapshots[ep]
        pos = np.asarray(res.positions, dtype=np.float32)
        if len(pos) == 0:
            continue

        # Relative Darstellung zum Startpunkt, damit die Form besser vergleichbar wird.
        rel = pos - res.start
        start = np.array([0.0, 0.0], dtype=np.float32)
        c1 = None if res.corner1 is None else np.asarray(res.corner1, dtype=np.float32) - res.start
        c2 = None if res.corner2 is None else np.asarray(res.corner2, dtype=np.float32) - res.start

        ax.plot(rel[:, 0], rel[:, 1], linewidth=2.0, alpha=0.95)
        ax.scatter(start[0], start[1], s=70, marker="s", zorder=4, label="Start")
        ax.scatter(rel[-1, 0], rel[-1, 1], s=45, marker="o", zorder=4, label="Ende")

        if c1 is not None:
            ax.scatter(c1[0], c1[1], s=65, marker="^", zorder=5, label="Ecke 1")
        if c2 is not None:
            ax.scatter(c2[0], c2[1], s=65, marker="x", zorder=5, label="Ecke 2")

        # Dreieck gestrichelt anzeigen, wenn beide Ecken existieren.
        if c1 is not None and c2 is not None:
            tri = np.vstack([start, c1, c2, start])
            ax.plot(tri[:, 0], tri[:, 1], linestyle="--", linewidth=1.5, alpha=0.7)

        # Quadratische, gepolsterte Achsen statt zusammengequetschter extremer Aspect-Ratios.
        pts = [rel]
        pts.append(start.reshape(1, 2))
        if c1 is not None:
            pts.append(c1.reshape(1, 2))
        if c2 is not None:
            pts.append(c2.reshape(1, 2))
        all_pts = np.vstack(pts)
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        span = max(xmax - xmin, ymax - ymin, 0.5)
        pad = 0.18 * span
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * span + pad
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x relativ zum Start")
        ax.set_ylabel("y relativ zum Start")
        ax.set_title(f"Ep {ep}\nR={res.reward:.2f}, A={res.area:.2f}, Erfolg={res.success}")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)))

    fig.suptitle("Wie sich die Sequenzen des Agenten über die Zeit verändern", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
