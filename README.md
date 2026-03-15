# RL-Exam

Algorithm: Actor–Critic with Antagonist
This task models an anthropomorphic index finger as a planar three-joint robot with fixed link lengths, 5 cm, 2.5 cm and 2.5 cm. 
All joints have a mobility of +/- 90°. Your objective is to learn a policy that generates a closed trajectory forming the largest possible triangle within the reachable workspace.
Joint limits, kinematic constraints, and disturbances must be respected. An antagonist should introduce bounded perturbations to test robustness.
You should clearly explain how the triangle is identified, how its area is computed, and how this objective is reflected in the reward function.

Evaluation criteria – Coding assignment:
0. Environment and algorithm uploaded and executable
1. Clear summary of the problem and solution approach
2. Demonstration of environment behaviour with example rollouts
3. Reward function definition and justification
4. Definition of test scenarios prior to training
5. Code documentation and structure
6. Demonstration of good policy performance
7. Quantitative figures of merit and hyperparameter studies
8. Ability to modify code live and explain performance changes

Important:
• You must implement your own environment in Python: Define what the actions and what the states are.
• Your code must be runnable independently by the examiners.
• You will be asked to explain, modify, and re-run parts of your code during the oral exam.
Deliverables:
• Source code (environment, algorithm, training, evaluation)
• Presentation slides (coding task + assigned scientific paper)
• All materials must be uploaded before the examination.

Funktionalität der Umgebung:
    - Finger mit drei Gelenken
        -> Verbindungsstücke haben Länge 5 cm, 2.5 cm und 2.5 cm
        -> Gelenke haben alle eine mögliche Range von +/- 90 Grad
    - durch Bewegung des Fingers soll ein Dreieck identifiziert werden
    - Die Fläche wird berechnet
        -> Brute-Force Bildung aller möglichen Dreiecke
    - die Fläche beeinflusst dann die reward function
    - Abgrebrochen wird eine Sequenz bei annäherung des Fingers auf eine Einheit zum Startpunkt nach einer Mindestanzahl an Schritten oder bei Überschreitung der maximalen Schrittanzahl
    - ein optionaler Antagonist fügt Störungen in die Bewegung des Fingers ein
    - Antagonist wirkt als Störung (Simulation realistischer Umstände) im Rahmen von bounded Störungen (nur begrenzt)
Reward-Funktion:
    - 3 Phasen
        -> 1. Phase: Weg vom Start und klare Ecke erkennbar
        -> 2. Phase: Weg von Ecke und klare Ecke erkennbar
        -> 3. Phase: Zurück zum Start
    - wichtiger Faktor: Fläche des identifizierten Dreiecks
        -> bei keinem Dreieck: Reward = 0
    - Reward bei Abschließen in der Nähe des Anfangspunktes
    - Penalty bei:
        -> überdrehen eines Gelenks
        -> keinem identifizierten Dreieck
        -> optional/zukünftig: zu starker Abweichung von gerader Linie

Verbesserungen, die das Training deutlich effektiver gemacht haben/Erkenntnisse:
    Environment:

    Agent:

Probleme/Herausforderungen:
    - Agenten früh dazu kriegen, die rewards zu erkennen (Ecken bilden und Dreieck zuende bringen)
    - Rewardfunction so gewichten, dass das richtige gelernt wird

Project Summary

This project implements a custom reinforcement learning environment in which the fingertip of a planar three-joint finger should generate a closed trajectory that resembles a triangle with large area. The finger has three links of length 5 cm, 2.5 cm, and 2.5 cm, and every joint is limited to +/- 90 degrees. The task is challenging because the agent must simultaneously learn meaningful corners, straight segments, sufficient enclosed area, and closure back to the start.

Initial implementation
    - A custom Gymnasium environment was created with discrete actions for changing the three joint angles.
    - A first actor-critic implementation was used together with a phase-based reward:
        -> phase 1: move away from the start
        -> phase 2: form the second edge / corner
        -> phase 3: return to the start
    - The environment tracked start position, fingertip trajectory, corners, triangle area, and optional antagonist disturbances.

First observations
    - Early training showed that reward could improve while success stayed low.
    - The agent often learned partial progress, such as moving outward or finding one corner, without reliably completing a valid triangular closed trajectory.
    - The final plots often did not look triangular, which revealed that the reward and the actual task objective were not fully aligned.

Main debugging and improvement steps
    1. Reward shaping analysis
        - The first reward design over-valued partial progress and under-valued successful completion.
        - Corner detection and turn penalties partly worked against each other.
        - The success threshold was also strict enough that some nearly valid trajectories were still counted as failures.

    2. Environment reward tuning
        - Corner rewards and closure rewards were rebalanced.
        - Phase-2 return-to-start shaping was strengthened.
        - Reward scale was reduced later because very large terminal rewards destabilized learning.

    3. Better diagnostics
        - Training logs were extended with:
            -> whether corner 1 was found
            -> whether corner 2 was found
            -> whether phase 2 was reached
            -> distance to start
        - This made it possible to see whether the agent was failing at corner detection, triangle formation, or closure.

    4. Objective clarification
        - A key conceptual clarification was that the task should not only identify a triangle from some visited points.
        - Instead, the fingertip trajectory itself should be triangular:
            -> three approximately straight edges
            -> two meaningful corners before closure
            -> no excessive extra corners or curvature
        - Based on this, new shape-quality metrics were introduced:
            -> segment straightness
            -> mean line deviation
            -> extra-corner counting

    5. Algorithm upgrade
        - The original return-based actor-critic was replaced by a GAE-based actor-critic.
        - Generalized Advantage Estimation made training much more stable and reduced variance in the policy updates.

    6. Benchmarking with PPO
        - PPO was used as a comparison benchmark.
        - PPO also struggled with the stricter trajectory-is-triangle objective.
        - This showed that the remaining difficulty was not only due to the actor-critic algorithm, but also due to the genuine hardness of the environment.

    7. Curriculum learning
        - The strongest improvement came from curriculum learning.
        - Instead of training immediately on the hardest version of the task, the environment was split into progressively stricter stages:
            -> easier closure and looser shape constraints first
            -> then increasingly stricter closure, area, and straightness requirements
        - A 4-stage curriculum was introduced in the final version.
        - This significantly improved early and mid-stage learning and reduced collapse when moving to harder settings.

Current state of the project
    - The project now consists of:
        -> a custom finger environment
        -> a GAE-based actor-critic trainer
        -> curriculum learning across multiple task difficulties
        -> reporting for reward, area, success, straightness, and extra corners
        -> comparison tooling for PPO and random baselines
    - The agent can learn meaningful triangular behavior in easier and medium stages and retains part of that behavior in the hardest stage.
    - The strictest final setting is still the main bottleneck, especially for keeping extra corners low and maintaining closure quality.

Most important lessons learned
    - Improving reward alone is not enough if the reward does not match the real task objective.
    - A better-aligned objective made the problem harder, but also made the results more meaningful.
    - GAE improved training stability.
    - Curriculum learning was the most effective method for improving learning performance.
    - The hardest stage still remains challenging, but performance no longer collapses completely.

Recommended presentation focus
    - Start from the original problem definition and explain why the task is difficult.
    - Show the mismatch between early reward improvement and low real success.
    - Explain how the objective was refined from “triangle can be found” to “trajectory should itself be triangular”.
    - Present the transition from simple actor-critic to GAE actor-critic.
    - Highlight curriculum learning as the main breakthrough.
    - Use the final visualizations to show best successful trajectories, largest-area trajectories, and best triangle-shape trajectories.
