import random
from dataclasses import dataclass

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


def discounted_returns(rewards: list[float], gamma: float, normalize: bool = False) -> torch.Tensor:
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)

    if normalize and len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    return returns_t


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
    antagonist_prob: float,
):
    obs, _ = env.reset()

    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[float] = []

    done = False
    truncated = False

    while not (done or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        protagonist_action = dist.sample()

        action = {
            "protagonist": int(protagonist_action.item()),
            "antagonist": choose_antagonist_action(antagonist_prob),
        }

        next_obs, reward, done, truncated, _ = env.step(action)

        log_probs.append(dist.log_prob(protagonist_action).squeeze())
        values.append(value.squeeze())
        entropies.append(dist.entropy().squeeze())
        rewards.append(float(reward))

        obs = next_obs

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

    returns = discounted_returns(rewards, gamma, normalize=False)
    log_probs_t = torch.stack(log_probs)
    values_t = torch.stack(values)
    entropies_t = torch.stack(entropies)

    return result, log_probs_t, values_t, returns, entropies_t


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


def plot_episode(ax, result: EpisodeResult):
    ax.set_title(
        f"Ep {result.episode}\n"
        f"R={result.reward:.2f}, A={result.area:.2f}, Erfolg={result.success}"
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

    ax.axhline(0, linewidth=0.6, alpha=0.4)
    ax.axvline(0, linewidth=0.6, alpha=0.4)
    ax.set_xlabel("x relativ zum Start")
    ax.set_ylabel("y relativ zum Start")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def plot_results(reward_history, area_history, success_history, actor_loss_history, critic_loss_history, snapshots):
    reward_ma = moving_average(reward_history, 25)
    area_ma = moving_average(area_history, 25)
    success_ma = moving_average(success_history, 25) * 100.0
    actor_loss_ma = moving_average(actor_loss_history, 25)
    critic_loss_ma = moving_average(critic_loss_history, 25)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Wie sich die Sequenzen des Actor-Critic-Agenten über die Zeit verändern", fontsize=18)

    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(reward_history, alpha=0.25, label="Reward pro Episode")
    ax1.plot(reward_ma, linewidth=2.5, label="Moving Average (25)")
    ax1.set_title("Reward über die Zeit")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(area_history, alpha=0.25, label="Fläche pro Episode")
    ax2.plot(area_ma, linewidth=2.5, label="Moving Average (25)")
    ax2.set_title("Dreiecksfläche über die Zeit")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Fläche")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(success_ma, linewidth=2.5)
    ax3.set_title("Success-Rate (Moving Average 25)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Erfolgsquote in %")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(actor_loss_history, alpha=0.25, label="Actor-Loss")
    ax4.plot(actor_loss_ma, linewidth=2.5, label="Moving Average (25)")
    ax4.plot(critic_loss_history, alpha=0.25, label="Critic-Loss")
    ax4.plot(critic_loss_ma, linewidth=2.5, label="Critic MA (25)")
    ax4.set_title("Training-Losses")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Loss")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    ordered_eps = sorted(snapshots.keys())[:8]
    for idx, ep in enumerate(ordered_eps):
        row = 1 + idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        plot_episode(ax, snapshots[ep])

    handles, labels = fig.axes[4].get_legend_handles_labels() if len(fig.axes) > 4 else ([], [])
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4)

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

    # Hyperparameter
    episodes = 600
    gamma = 0.99
    learning_rate = 1e-3
    critic_weight = 0.5
    entropy_weight = 0.01
    antagonist_prob = 0.10
    use_antagonist = True

    env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
    obs_dim = int(np.prod(env.observation_space.shape))

    model = ActorCriticNet(obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reward_history: list[float] = []
    area_history: list[float] = []
    success_history: list[int] = []
    actor_loss_history: list[float] = []
    critic_loss_history: list[float] = []
    snapshots: dict[int, EpisodeResult] = {}

    snapshot_episodes = {1, 10, 50, 100, 200, 500, episodes}

    print("Starte Actor-Critic-Lauf ...")
    print("Der Actor lernt die Policy, der Critic lernt den Zustandswert V(s).")
    print("Optimiert wird: Actor-Loss + Critic-Loss - Entropiebonus.\n")

    for ep in range(1, episodes + 1):
        result, log_probs, values, returns, entropies = run_episode(
            env=env,
            model=model,
            gamma=gamma,
            antagonist_prob=antagonist_prob,
        )
        result.episode = ep

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = entropies.mean()

        loss = actor_loss + critic_weight * critic_loss - entropy_weight * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        result.actor_loss = float(actor_loss.item())
        result.critic_loss = float(critic_loss.item())
        result.entropy = float(entropy_bonus.item())

        reward_history.append(result.reward)
        area_history.append(result.area)
        success_history.append(int(result.success))
        actor_loss_history.append(result.actor_loss)
        critic_loss_history.append(result.critic_loss)

        if ep in snapshot_episodes:
            snapshots[ep] = result

        if ep % 50 == 0 or ep == 1:
            avg_reward = float(np.mean(reward_history[-50:]))
            avg_area = float(np.mean(area_history[-50:]))
            success_rate = float(np.mean(success_history[-50:]) * 100.0)
            avg_actor_loss = float(np.mean(actor_loss_history[-50:]))
            avg_critic_loss = float(np.mean(critic_loss_history[-50:]))
            print(
                f"Episode {ep:>3d} | "
                f"Reward={result.reward:>8.3f} | "
                f"Area={result.area:>6.3f} | "
                f"Success={str(result.success):>5s} | "
                f"ActorLoss={result.actor_loss:>8.3f} | "
                f"CriticLoss={result.critic_loss:>8.3f} | "
                f"letzte50: meanReward={avg_reward:>8.3f}, meanArea={avg_area:>6.3f}, "
                f"success={success_rate:>5.1f}%, meanActorLoss={avg_actor_loss:>8.3f}, meanCriticLoss={avg_critic_loss:>8.3f}"
            )

    plot_results(
        reward_history=reward_history,
        area_history=area_history,
        success_history=success_history,
        actor_loss_history=actor_loss_history,
        critic_loss_history=critic_loss_history,
        snapshots=snapshots,
    )


if __name__ == "__main__":
    main()
