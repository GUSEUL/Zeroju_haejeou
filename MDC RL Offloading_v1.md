# MDC RL Offloading Project Documentation

This project implements a Reinforcement Learning (RL) framework for Mobile Distributed Computing (MDC) task offloading. It includes a custom Gymnasium environment, multiple baseline agents, and advanced visualization tools.

## 1. Technical Specifications

### State Space ($\mathcal{S}$)
The observation space is defined as a discrete vector:
$S = [Task, Channel, CPU, Queue, BW]$
- **Task**: $\{0: \text{URLLC (Delay Sensitive)}, 1: \text{eMBB (Throughput Sensitive)}\}$
- **Channel**: $\{0: \text{Poor}, 1: \text{Normal}, 2: \text{Good}\}$
- **CPU**: $\{0: \text{Low Load (High Rate)}, 1: \text{Normal}, 2: \text{Congested (Low Rate)}\}$
- **Queue**: $\{0: \text{Empty}, 1: \text{Smooth}, 2: \text{Warning}, 3: \text{Critical}\}$
- **BW (Bandwidth)**: $\{0: \text{Insufficient}, 1: \text{Sufficient}\}$

### Action Space ($\mathcal{A}$)
The agent can choose from 8 discrete actions:
- $a = 0$: **Local Processing** (Process on the Main MDC)
- $a \in \{1, 2, 3, 4, 5, 6\}$: **Offload** to Neighboring MDC $N_1$ to $N_6$
- $a = 7$: **Intentional Drop** (Discard task to save energy/queue)

### Reward Function ($R$)
The reward is designed to balance latency, energy, and reliability:
$R = r_{\text{delay}} + p_{q} + r_{\text{energy}} + r_{\text{drop}}$

1. **Delay Penalty**: $r_{\text{delay}} = -(w_{\text{task}} \cdot \text{TotalDelay} \cdot 2.0)$
   - $w_{\text{task}} = 2.0$ for URLLC, $0.5$ for eMBB.
2. **Queue Penalty**: $p_q = -(\exp(2.0 \cdot \frac{Queue}{3}) - 1)$
3. **Energy Penalty**: $r_{\text{energy}} = -(\text{EnergyConsumed} \cdot 1.5)$
4. **Drop Penalty**: $r_{\text{drop}} = -20.0$ if task is dropped.

---

## 2. File Descriptions & Role

### Core Environment
- **`mdc_gym_env.py`**: The primary Gymnasium environment. Defines physical dynamics, service rates, and traffic arrival (Poisson/Pareto).
- **`unstable_mdc_env.py`**: A subclass of the main environment with higher stochasticity in channel quality and bandwidth for stress testing.

### Training & Evaluation
- **`compare_baselines.py`**:
    - **Role**: Trains **SARSA** and **Q-Learning** agents and compares them with **All-Local**, **Random**, and **Threshold** baselines.
    - **Checkpoints**: Implements `save_q_table` and `load_q_table` to reuse pre-trained models.
    - **Output**: `final_comparison_results.csv`, `sarsa_q_table.npy`, `ql_q_table.npy`.
- **`compare_unstable_network.py`**:
    - **Role**: Evaluates agent robustness in the unstable environment.
    - **Output**: `unstable_network_comparison.csv`, `sarsa_q_table_unstable.npy`, `ql_q_table_unstable.npy`.

### Visualization
- **`visualize_results.py`**:
    - **Role**: Generates static performance plots and Q-table heatmaps showing the learned policy across CPU/Queue states.
    - **Output**: `performance_comparison_all.png`, `policy_heatmap_sarsa.png`, `policy_heatmap_q-learning.png`.
- **`visualize_network.py`**:
    - **Role**: High-fidelity Pygame simulation. Shows real-time packet movement, queue levels (Green/Orange/Red), and agent decisions.
    - **Visual Cues**: 
        - **Orange**: Local processing (Internal pulse).
        - **Blue**: Offloading to neighbors.
        - **Red Flash**: Task dropped.

### Automation
- **`run_all_mdc.ps1`**: PowerShell script to execute the entire pipeline (Compare -> Unstable -> Visualize) in sequence.

---

## 3. Implementation Logic

### Checkpoint Logic
To avoid redundant training, each script utilizes the following logic:
```python
def train_sarsa(env, episodes=5000, checkpoint="sarsa_q_table.npy"):
    q_table = load_q_table(checkpoint)
    if q_table is not None:
        return q_table, 0.0 # Skip training if file exists
    # ... training code ...
    save_q_table(q_table, checkpoint)
```
Checkpoints are stored in the same directory as the scripts for consistency.
