# GridWorld MDP Visualizer — Value Iteration & Policy Iteration (Streamlit)

**Interactive web-based GridWorld Markov Decision Process (MDP) visualizer** to study optimal policy emergence using **Value Iteration** and **Policy Iteration** algorithms. Features stochastic transitions, terminal states, obstacles, and step-by-step iteration visualization.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org)

## Live Demo (Deployed)
🔗 **[Streamlit App](https://mdp-visualizer.streamlit.app/)**

---

###  GridWorld MDP Environment
-  5x5 Grid Layout
-  3 Obstacles: (1,1), (2,1), (3,1)
-  Goal: (0,4) [+10 reward]
-  Negative Terminal: (1,3) [-10 reward]
- Start: (4,0)

### Stochastic Transitions
- **80%** intended direction (p_intended = 0.8)
- **20%** slip probability distributed across other 3 directions

### Rewards Structure
| Transition | Reward |
|------------|---------|
| Reach Goal | **+10** |
| Reach Negative Terminal | **-10** |
| Each Step | **-0.1** |

### Visualizations
- **Value Function**: Heatmap + numeric values per cell
- **Policy**: Directional arrows (↑↓←→)
- **Iteration Playback**: Step-by-step algorithm execution
- **Convergence Status**: Progress bar + iteration counter

### Interactive Controls
- **Algorithm**: Value Iteration ↔ Policy Iteration
- **γ Slider**: Discount factor (0.01-0.99)
- **θ Input**: Convergence threshold
- **Navigation**: Start/Stop/Back/Next/End/Reset

---

## Dashboard Interface
![Dashboard](https://github.com/MaidaKosser/MDP-GridWorld/blob/main/dashboard.png)

## Project Structure
```bash
MDP-GridWorld/
├── app.py                  # Streamlit UI entry point
├── src/
│   ├── mdp_gridworld.py     # MDP definition: states/actions/transitions/rewards
│   ├── dp_algorithms.py     # Value Iteration + Policy Iteration
│   └── render.py            # Heatmap + arrows visualization
├── requirements.txt
└── README.md
```

## Installation & Setup
1) Clone the Repository
```bash
git clone https://github.com/MaidaKosser/MDP-GridWorld.git
cd MDP-GridWorld
```
2) Create Virtual Environment (Recommended)
```bash
python -m venv .venv
```
3) Install Dependencies
```bash
pip install -r requirements.txt
streamlit run app.py
```

- Open the provided local URL (usually http://localhost:8501) in your browser.

## How to Use (Step-by-Step)
- Select an algorithm: Value Iteration or Policy Iteration

- Adjust γ (discount factor) and θ (convergence threshold)

Use controls:

- Start: auto-play iterations

- Stop: pause

- Back / Next: step-by-step navigation

- End: jump to final converged iteration

- Reset: recompute history for current parameters

- Comparison of convergence speed: Value Iteration vs Policy Iteration.

