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
    - wichtiger Faktor: Fläche des identifizierten Dreiecks
        -> bei keinem Dreieck: Reward = 0
    - Penalty bei:
        -> überdrehen eines Gelenks
        -> keinem identifizierten Dreieck
        -> optional/zukünftig: zu starker Abweichung von gerader Linie

Verbesserungen, die das Training deutlich effektiver gemacht haben/Erkenntnisse:
    Environment:
        - Evaluation der gemalten Fläche nach jedem Schritt, nicht erst nach Episodenende
    Agent: