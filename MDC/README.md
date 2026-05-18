# MDC RL Offloading Project

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

---

## 2. File Descriptions & Role

### Core Environment
- **`mdc_gym_env.py`**: The primary Gymnasium environment.
    - **Configurable Load**: Supports `arrival_lambda` to control the Poisson arrival rate of tasks.
    - **Neighbor Queues**: Tracks 6 individual neighbor queues.
- **`unstable_mdc_env.py`**: A subclass for stress testing with high stochasticity.

### Training & Evaluation
- **`compare_baselines.py`**:
    - **Role**: Trains **SARSA** and **Q-Learning** agents and compares them with baselines.
    - **Versioned Checkpoints**: Saves Q-tables with lambda suffixes (e.g., `sarsa_q_table_L1.5.npy`).
    - **Usage**: `python compare_baselines.py --lambda_val 1.5 --episodes 10000`
- **`save_state_values.py`**:
    - **Role**: Extracts and calculates state values ($V(s) = \max_a Q(s,a)$) for all possible state configurations and saves them to a CSV file for diagnostic analysis.
    - **Usage**: `python save_state_values.py --lambda_val 1.5 --agent sarsa`

### Visualization
- **`visualize_results.py`**:
    - **Role**: Generates performance bar charts, training convergence curves, and policy heatmaps.
    - **Usage**: `python visualize_results.py --lambda_val 1.5`
- **`visualize_training_metrics.py`**:
    - **Role**: Specialized script to plot detailed **Reward** and **Drop Count** trends over training episodes.
    - **Usage**: `python visualize_training_metrics.py --lambda_val 1.5`
- **`visualize_network.py`**:
    - **Role**: High-fidelity Pygame simulation with real-time feedback.
    - **Enhanced Animations**: Features distinct visual cues for local processing (Orange), offloading (Blue), and drops (Red).
    - **Usage**: `python visualize_network.py --lambda_val 1.5`

---

## 3. Implementation Logic

### Traffic Load (Lambda)
The environment uses a Poisson process for task arrivals. By adjusting the `--lambda_val` argument:
- **Lambda < 1.0**: Low load, local processing is dominant.
- **Lambda 1.5 - 2.0**: Normal/High load, offloading strategy is critical.
- **Lambda > 2.5**: Extreme load, tests robustness and drop mitigation.

### Checkpoint & Log Management
To ensure results from different stress tests are preserved, all output files follow a versioned naming convention:
- **Checkpoints**: `{agent}_q_table_L{lambda}.npy`
- **Logs**: `{agent}_train_log_L{lambda}.csv`
- **Plots**: `{metric}_L{lambda}.png`

### Getting Started
1. Install dependencies: `pip install gymnasium pygame numpy pandas matplotlib seaborn`
2. Train agents: `python compare_baselines.py --lambda_val 1.5`
3. (Optional) Extract state values: `python save_state_values.py --lambda_val 1.5 --agent sarsa`
4. Visualize results: `python visualize_results.py --lambda_val 1.5`
5. Run simulation: `python visualize_network.py --lambda_val 1.5`
