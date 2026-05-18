import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mdc_gym_env import MDCOffloadingEnv
from compare_baselines import train_sarsa, train_q_learning, evaluate, AllLocalAgent, RandomAgent, ThresholdAgent

def plot_performance_comparison(results):
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
    plt.savefig('performance_comparison_all.png')
    print("Comprehensive performance plot saved as 'performance_comparison_all.png'")

def plot_q_heatmap(q_table, env, title_suffix="Agent"):
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
    plt.title(f'Learned Policy Heatmap ({title_suffix})\n(0:Local, 1-6:Offload, 7:Drop)')
    plt.xlabel('Queue Level')
    plt.ylabel('CPU Workload')
    plt.savefig(f'policy_heatmap_{title_suffix.lower()}.png')
    print(f"Policy heatmap saved as 'policy_heatmap_{title_suffix.lower()}.png'")

if __name__ == "__main__":
    env = MDCOffloadingEnv()
    
    print("Training agents for visualization...")
    sarsa_q, _ = train_sarsa(env, episodes=3000)
    ql_q, _ = train_q_learning(env, episodes=3000)
    
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
        
    plot_performance_comparison(results)
    plot_q_heatmap(sarsa_q, env, title_suffix="SARSA")
    plot_q_heatmap(ql_q, env, title_suffix="Q-Learning")
