import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mdc_gym_env import MDCOffloadingEnv
from compare_baselines import train_sarsa, train_q_learning, evaluate, AllLocalAgent, RandomAgent, ThresholdAgent

def plot_performance_comparison(results, lambda_val=None):
    """Generates bar charts for Reward, Drop Rate, and Energy for all agents."""
    labels = list(results.keys())
    rewards = [results[l][0] for l in labels]
    drop_rates = [results[l][1] for l in labels]
    energies = [results[l][2] for l in labels]

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Rewards
    ax[0].bar(labels, rewards, color='skyblue')
    ax[0].set_title('Average Reward (Higher is Better)')
    ax[0].set_ylabel('Reward')

    # 2. Drop Rates
    ax[1].bar(labels, drop_rates, color='salmon')
    ax[1].set_title('Drop Rate (%) (Lower is Better)')
    ax[1].set_ylabel('Percentage (%)')

    # 3. Energy Consumption
    ax[2].bar(labels, energies, color='lightgreen')
    ax[2].set_title('Avg Energy Consumption (J) (Lower is Better)')
    ax[2].set_ylabel('Energy (Joules)')

    plt.tight_layout()
    suffix = f"_L{lambda_val}" if lambda_val else ""
    filename = f'performance_comparison_all{suffix}.png'
    plt.savefig(filename)
    print(f"Comprehensive performance plot saved as '{filename}'")

def plot_training_curves(lambda_val=None):
    """Plots training reward curves from saved logs."""
    suffix = f"_L{lambda_val}" if lambda_val else ""
    sarsa_log = f"sarsa_train_log{suffix}.csv"
    ql_log = f"q_learning_train_log{suffix}.csv"
    
    plt.figure(figsize=(12, 6))
    
    try:
        if os.path.exists(sarsa_log):
            df_sarsa = pd.read_csv(sarsa_log)
            # Rolling average for smoother curve
            plt.plot(df_sarsa['reward'].rolling(window=100).mean(), label='SARSA', color='blue', alpha=0.8)
        
        if os.path.exists(ql_log):
            df_ql = pd.read_csv(ql_log)
            plt.plot(df_ql['reward'].rolling(window=100).mean(), label='Q-Learning', color='green', alpha=0.8)
            
        plt.title(f'Training Convergence (Lambda={lambda_val if lambda_val else "Default"})')
        plt.xlabel('Episode')
        plt.ylabel('Rolling Average Reward (100 eps)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        filename = f'training_curves{suffix}.png'
        plt.savefig(filename)
        print(f"Training curves plot saved as '{filename}'")
    except Exception as e:
        print(f"Error plotting training curves: {e}")

def plot_q_heatmap(q_table, env, title_suffix="Agent", lambda_val=None):
    heatmap_data = np.zeros((3, 4)) 
    for cpu in range(3):
        for queue in range(4):
            state = np.array([0, 1, cpu, queue, 1])
            idx = env.get_state_index(state)
            heatmap_data[cpu, queue] = np.argmax(q_table[idx])
            
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", 
                xticklabels=['Empty', 'Smooth', 'Warning', 'Critical'],
                yticklabels=['Low', 'Normal', 'Congested'])
    plt.title(f'Learned Policy Heatmap ({title_suffix}, L={lambda_val})\n(0:Local, 1-6:Offload, 7:Drop)')
    plt.xlabel('Queue Level')
    plt.ylabel('CPU Workload')
    
    suffix = f"_L{lambda_val}" if lambda_val else ""
    filename = f'policy_heatmap_{title_suffix.lower()}{suffix}.png'
    plt.savefig(filename)
    print(f"Policy heatmap saved as '{filename}'")

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=None, help="Poisson arrival lambda")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    args = parser.parse_args()

    env = MDCOffloadingEnv(arrival_lambda=args.lambda_val)
    
    print(f"Loading/Training agents for visualization (Lambda={args.lambda_val})...")
    # Using the same logic as compare_baselines to load/train
    sarsa_q, _ = train_sarsa(env, episodes=args.episodes)
    ql_q, _ = train_q_learning(env, episodes=args.episodes)
    
    agents = {
        "SARSA": (sarsa_q, True),
        "Q-Learning": (ql_q, True),
        "All-Local": (AllLocalAgent(), False),
        "Random": (RandomAgent(), False),
        "Threshold": (ThresholdAgent(threshold=2), False)
    }
    
    results = {}
    print("\nEvaluating all agents...")
    for name, (agent, is_q) in agents.items():
        avg_r, drop_pct, avg_e = evaluate(env, agent, is_q, name)
        results[name] = (avg_r, drop_pct, avg_e)
        
    plot_performance_comparison(results, lambda_val=args.lambda_val)
    plot_training_curves(lambda_val=args.lambda_val)
    plot_q_heatmap(sarsa_q, env, title_suffix="SARSA", lambda_val=args.lambda_val)
    plot_q_heatmap(ql_q, env, title_suffix="Q-Learning", lambda_val=args.lambda_val)
