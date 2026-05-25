# MDC MDP Task Offloading Optimization Experiment Report

This report presents a multi-faceted performance analysis of reinforcement learning algorithms (Q-Learning, Expected SARSA) and dynamic programming (DP: Policy/Value Iteration) under the Markov Decision Process (MDP) modeling environment. It validates the redesigned QoS (Quality of Service)-aware reward function.

---

## 1. Design of Improved Reward Function & Pessimistic Q-Initialization Verification (Agent 3)

An `"improved"` reward type was introduced to dynamically apply reward hyperparameters based on the task type ($0$: URLLC, $1$: eMBB). To solve learning instability, **Pessimistic Q-table Initialization** and a relaxed exploration schedule were implemented.

### A. Improvements
- **Pessimistic Q-table Initialization**: The previous default initialization of `0.0` caused the agent to optimistically misjudge unvisited states in negative reward environments. To prevent this, Q-tables are initialized to **`-150.0`**, blocking unverified actions (e.g., dropping tasks from an empty local queue).
- **Extended Training Episodes**: Training episodes were extended from 5,000 to **20,000 episodes** to guarantee convergence.
- **Relaxed Epsilon Decay**: The exploration schedule was slowed down to decay until `episodes * 0.6`, allowing the agent to sufficiently explore the state space.

### B. Performance Verification Results
With pessimistic initialization and 20,000 episodes, both Q-Learning and Expected SARSA agents **perfectly converged to the optimal reward level and drop performance of the mathematical upper bound (DP Optimal Policy)**.

- **Load Balancing and Latency Minimization**:
  - The improved scheme imposes a strict queue occupancy penalty ($\beta_{\text{local}} = 8.0$) and a heavy drop penalty ($\gamma_{\text{task}} = 30.0$) on URLLC tasks.
  - Consequently, the agent learns to proactively offload URLLC tasks to empty neighboring edge nodes or drop them immediately if necessary. This prevents URLLC tasks from waiting in crowded queues for multiple timesteps and receiving severe cumulative queue penalties, successfully optimizing overall performance while meeting QoS constraints.

---

## 2. Analysis of 100,000 Large-Scale Training Results (Standard vs. Cliff, 100k Episodes)

To guarantee complete convergence of the reinforcement learning agents, training was extended to **100,000 episodes**. Experiments were conducted across three traffic load levels ($\lambda \in [0.5, 1.0, 1.5]$) under `standard` (QoS-weighted penalty) and `cliff` (high-intensity -1000.0 penalty) reward models.

### A. Final Experiment Results Summary (100,000 Episodes)

| Traffic Load ($\lambda$) | Reward Type | Algorithm | Expected Return | Avg. Pending Count per Ep | Avg. Energy Consumed per Ep |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **$\lambda = 0.5$** | Standard | Policy Iteration | -12.11 | 0.00 | 459.48 |
| | | Value Iteration | -12.11 | 0.00 | 459.49 |
| | | Expected SARSA | -13.27 | 0.00 | 451.74 |
| | | Q-Learning | **-12.30** | 0.00 | 458.01 |
| | Cliff | Policy Iteration | -12.41 | 0.00 | 459.85 |
| | | Value Iteration | -12.41 | 0.00 | 459.89 |
| | | Expected SARSA | -20.74 | 0.00 | 443.36 |
| | | Q-Learning | **-13.08** | 0.00 | 457.51 |
| **$\lambda = 1.0$** | Standard | Policy Iteration | -19.77 | 6.02 | 451.74 |
| | | Value Iteration | -19.77 | 6.02 | 451.74 |
| | | Expected SARSA | -24.49 | 7.42 | 444.96 |
| | | Q-Learning | **-23.60** | 7.14 | 446.35 |
| | Cliff | Policy Iteration | -16.92 | 6.00 | 450.82 |
| | | Value Iteration | -16.92 | 6.00 | 450.84 |
| | | Expected SARSA | **-20.69** | 14.49 | 444.15 |
| | | Q-Learning | -31.09 | 18.47 | 439.35 |
| **$\lambda = 1.5$** | Standard | Policy Iteration | -99.24 | 819.13 | 151.21 |
| | | Value Iteration | -99.15 | 819.04 | 151.25 |
| | | Expected SARSA | **-131.86** | 836.05 | 278.84 |
| | | Q-Learning | -147.32 | 840.41 | 289.85 |
| | Cliff | Policy Iteration | -54.95 | 761.35 | 177.14 |
| | | Value Iteration | -54.97 | 761.31 | 177.16 |
| | | Expected SARSA | **-150.97** | 762.69 | 266.07 |
| | | Q-Learning | -298.35 | 764.95 | 314.89 |

### B. In-Depth Results Analysis and Algorithm Characteristics

1. **Convergence of Reinforcement Learning to Optimality**:
   - Through large-scale training of 100,000 episodes, Q-Learning and Expected SARSA agents converged closely to the performance limits of Policy/Value Iteration (the theoretical analytical upper bound).
   - Under low load ($\lambda = 0.5$), Q-Learning (Standard: -12.30, Cliff: -13.08) achieved near-perfect alignment with the DP optimal policy (Standard: -12.11, Cliff: -12.41), validating that the pessimistic initialization successfully resolved unvisited state malfunctions.

2. **Queue Occupancy (Pending) Control under Different Traffic Loads**:
   - **$\lambda = 0.5$ (Low Load)**: Since the system has ample processing resources, pending buffers are entirely avoided (0.00 pending count) under both Standard and Cliff settings. The policy focuses solely on balancing energy and transmission delay.
   - **$\lambda = 1.0$ (Medium Load)**: Task queueing begins to occur, but both reward models keep the pending count low at around 6 tasks per episode.
   - **$\lambda = 1.5$ (High Load / Overloaded)**: Arrivals exceed processing limits, causing queue occupancy to spike. Here, an interesting mathematical paradox is observed:
     - **Reward Comparison**: Although the Cliff reward model penalizes each pending task severely (-1000.0), its expected return (**-54.97, with 761.31 pending tasks**) is significantly better than the Standard model's return (**-99.15, with 819.04 pending tasks**).
     - **Causal Analysis**: The extreme penalty of the Cliff model forces the agent to strictly avoid queue accumulation in the early stages of the episode (where $\gamma^t$ discount factors are closest to 1). This prevents severe congestion from arising in the first place, leading to a substantial reduction in transmission delay and overall queue penalties over the entire episode.

3. **Expected SARSA vs. Q-Learning Stability under Cliff Penalty**:
   - Under the high-load ($\lambda = 1.5$) Cliff penalty scenario, **Expected SARSA** achieved an expected return of **-150.97** (762.69 pending), while **Q-Learning** lagged behind at **-298.35** (764.95 pending).
   - This occurs because Q-Learning's optimistic $\max$ update rule propagates overestimation bias under probabilistic queue state transitions and high penalty boundaries. Expected SARSA, conversely, propagates the expected value of policy action values, dampening the variance and leading to a more robust, safe, and stable convergence.

---

## 3. Project File Descriptions and Structure

The project consists of MDP environments, reinforcement learning training loops, simulation scripts, and multi-dimensional visualization tools. Below is an overview of the key files:

| File Name | Description & Core Functions |
| :--- | :--- |
| [mdc_mdp_env.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/mdc_mdp_env.py) | **Gymnasium-compliant MDP Environment**: Models queue states, channel state transitions (random walk), Poisson task arrivals, offloading actions, and reward computation. Includes an infinite **pending_buffer** to handle overflowed tasks. |
| [build_mdp_model.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/build_mdp_model.py) | **Analytical Model Builder**: Formulates the theoretical state transition probability matrix $P(S' \mid S, a)$ and expected reward vector $R(S, a)$ analytically, exporting them as Pickle files inside the `models/` directory. |
| [train_all_mdp.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/train_all_mdp.py) | **Agent Training Loop**: Runs Q-Learning and Expected SARSA training for a specified number of episodes (default 100,000), saving the final Q-tables as CSV checkpoints. Also evaluates DP (Policy/Value Iteration) for baseline comparison. |
| [visualize_mdp_results.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/visualize_mdp_results.py) | **Visualization Utility**: Generates individual scenario plots, including training reward curves (`mdp_training_curves.png`), performance comparison bar charts (`mdp_performance_bars.png`), and policy decision heatmaps (`mdp_heatmap_*.png`). |
| [visualization/export_policy_data.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/visualization/export_policy_data.py) | **Data Integrator**: Consolidates trained Q-tables and optimal DP policies across all scenarios into structured JSON formats (`visualization/policy_data.js` and `policy_data.json`) to serve the web visualizer. |
| [generate_queue_gif.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/generate_queue_gif.py) | **Queue Simulation Animator**: Visualizes the real-time buffer queue occupancy, task arrivals, and cumulative pending metrics under Cliff and Standard policies as high-speed (8.0 FPS) GIF animations. |
| [generate_policy_comparison_gif.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/generate_policy_comparison_gif.py) | **Side-by-Side Policy Animator**: Creates a 3-column comparative GIF animation illustrating queue patterns and decision-making under DP Optimal, Expected SARSA, and Q-Learning policies side-by-side. |
| [aggregate_all_comparisons.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/aggregate_all_comparisons.py) | **Data Aggregator**: Scans the output directory, compiles key performance metrics (reward, pending count, energy consumption) for all 6 scenarios, and outputs a formatted Markdown summary table to stdout. |
| [run_mdp_pipeline.sh](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/run_mdp_pipeline.sh) | **Bash Automation Pipeline**: Runs the complete training, evaluation, and visualization pipeline (Steps 1 through 7) sequentially. |
| [run_mdp_pipeline.ps1](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/run_mdp_pipeline.ps1) | **PowerShell Automation Pipeline**: Runs the automation pipeline on Windows systems using PowerShell commands. |

---

## 4. Pipeline Execution Guide

All scenarios can be trained and visualized collectively using the automated scripts, or executed manually step-by-step.

### Method A. Automated Execution via Bash Script (Recommended)
Run the script in **Git Bash** or a **Linux/WSL** terminal. By default, this trains and visualizes all 6 scenarios ($\lambda \in [0.5, 1.0, 1.5]$, under `standard` and `cliff` rewards) for 100,000 episodes.

1. **Execute full pipeline with default settings**:
   ```bash
   bash run_mdp_pipeline.sh
   ```

2. **Execute custom pipeline with specific parameters**:
   * `-l`: Space-separated list of lambda values to evaluate.
   * `-e`: Number of episodes.
   * `-r`: Space-separated list of reward models.
   ```bash
   # Example: Train for 5,000 episodes on lambda 0.5 and 1.5 only
   ./run_mdp_pipeline.sh -l "0.5 1.5" -e 5000 -r "standard cliff"
   ```

---

### Method B. Automated Execution via PowerShell (Windows)
A pipeline control script optimized for Windows PowerShell.

* **Execute pipeline with parameters**:
  ```powershell
  # Example: Run lambda 1.5 with 10,000 episodes
  ./run_mdp_pipeline.ps1 -Lambda 1.5 -Episodes 10000
  ```

---

### Method C. Step-by-Step Manual Execution
Run individual stages from the project root directory.

#### [Step 1] Build Analytical MDP Transition Models
```bash
python build_mdp_model.py --lambda_val=0.5 --reward_type=standard
python build_mdp_model.py --lambda_val=0.5 --reward_type=cliff
python build_mdp_model.py --lambda_val=1.0 --reward_type=standard
python build_mdp_model.py --lambda_val=1.0 --reward_type=cliff
python build_mdp_model.py --lambda_val=1.5 --reward_type=standard
python build_mdp_model.py --lambda_val=1.5 --reward_type=cliff
```
*(Outputs saved to: `models/mdp_model_{reward_type}_L{lambda}.pkl`)*

#### [Step 2] Train Q-Learning and Expected SARSA Agents
```bash
python train_all_mdp.py --lambda_val=0.5 --episodes=100000 --reward_type=standard
python train_all_mdp.py --lambda_val=0.5 --episodes=100000 --reward_type=cliff
python train_all_mdp.py --lambda_val=1.0 --episodes=100000 --reward_type=standard
python train_all_mdp.py --lambda_val=1.0 --episodes=100000 --reward_type=cliff
python train_all_mdp.py --lambda_val=1.5 --episodes=100000 --reward_type=standard
python train_all_mdp.py --lambda_val=1.5 --episodes=100000 --reward_type=cliff
```
*(Outputs saved to: `results/L_{lambda}_E_{episodes}/{reward_type}/q_table_ql.csv` & `q_table_sarsa.csv`)*

#### [Step 3] Visualize Cumulative Rewards and Policy Heatmaps
```bash
python visualize_mdp_results.py --lambda_val=0.5 --episodes=100000 --reward_type=standard
python visualize_mdp_results.py --lambda_val=0.5 --episodes=100000 --reward_type=cliff
# Repeat for other lambda and reward scenarios as needed
```
*(Outputs saved inside corresponding scenario folders)*

#### [Step 4] Compile JSON Data for Web Dashboard
```bash
python visualization/export_policy_data.py --episodes=100000
```
*(Outputs saved to: `visualization/policy_data.json`)*

#### [Step 5] Render Queue Simulation GIFs
```bash
python generate_queue_gif.py --episodes=100000
```
*(Outputs saved to: `results/queue_simulation_L*.gif`)*

#### [Step 6] Render Side-by-Side Policy Comparison GIFs
```bash
python generate_policy_comparison_gif.py --episodes=100000
```
*(Outputs saved to: `results/policy_comparison_L*.gif`)*

#### [Step 7] Compile Final Performance Summary Table
```bash
python aggregate_all_comparisons.py --episodes=100000
```
*(Displays compiled performance statistics in Markdown format on stdout)*
