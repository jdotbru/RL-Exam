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


Setup And Run:

To make the project executable on another computer, upload the full project folder with at least these files:
- `actor_critic_finger.py`
- `fingerTriangleEnv.py`
- `agent_helpers.py`
- `env_helpers.py`
- `plotting_helpers.py`
- `requirements.txt`
- `README.md`
- `test_environment.py`
- `Actions.txt` only as supplementary documentation

1. Python 3.10 oder neuer wird benötigt
2. alle Dateien bis auf "README.md", "requirements.txt" und "Actions.txt" auf die Endung ".py" ändern
3. Terminal im Projektverzeichnis öffnen
4. Virtuelle Umgebung erstellen:
   `python -m venv .venv`
5. Virtuelle Umgebung aktivieren:
   Windows: `.venv\Scripts\activate`
   macOS/Linux: `source .venv/bin/activate`
6. Dependencies installieren:
   `pip install -r requirements.txt`
7. Projekt laufen lassen:
   `python actor_critic_finger.py`
8. Environment Tests laufen lassen:
   `python test_environment.py`

What the script does:
- Starts training of the actor-critic agent
- Runs evaluations during and after training
- Writes a summary file `final_eval_summary.txt`
- Optionally writes `best_triangle_checkpoint.pt`
- Opens matplotlib plots at the end

Notizen:
- Alle Python Dateien müssen im gleichen Ordner sein, da sie sich gegenseitig importieren
- Falls die Plots nicht geöffnet werden, eventuell in einem externen Terminal ausführen
- Sollte nur eine kurze Demonstration gewünscht sein, kann die Episodenanzahl in`actor_critic_finger.py` niedriger gestellt werden
