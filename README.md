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

Setup And Run

To make the project executable on another computer, upload the full project folder with at least these files:
- `actor_critic_finger.py`
- `fingerTriangleEnv.py`
- `agent_helpers.py`
- `env_helpers.py`
- `plotting_helpers.py`
- `requirements.txt`
- `README.md`

Optional files:
- `best_triangle_checkpoint.pt` only if you want to provide a pretrained checkpoint
- `final_eval_summary.txt` only if you want to include the last recorded results
- `Actions.txt` only as supplementary documentation

Recommended steps for the other person:

1. Install Python 3.10 or newer.
2. Download or clone the project folder.
3. Open a terminal in the project directory.
4. Create a virtual environment:
   `python -m venv .venv`
5. Activate it:
   Windows: `.venv\Scripts\activate`
   macOS/Linux: `source .venv/bin/activate`
6. Install dependencies:
   `pip install -r requirements.txt`
7. Run the project:
   `python actor_critic_finger.py`

What the script does:
- Starts training of the actor-critic agent
- Runs evaluations during and after training
- Writes a summary file `final_eval_summary.txt`
- Optionally writes `best_triangle_checkpoint.pt`
- Opens matplotlib plots at the end

Notes for portability:
- Keep all Python files in the same folder, because they import each other via local file names.
- If plots do not open in the IDE, run the script from a normal terminal or configure a matplotlib backend supported on that system.
- On a machine without a GPU, PyTorch will still run on CPU, but training may take longer.
- If only a quick demonstration is needed, reduce the number of episodes in `actor_critic_finger.py`.
