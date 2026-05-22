import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def solve_dp(model_path):
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        P, R = pickle.load(f)
    n_states = P.shape[0]
    V = np.zeros(n_states)
    gamma = 0.95
    theta = 1e-12
    # Vectorized value iteration for 500x speedup
    for _ in range(10000):
        V_old = V.copy()
        Q = R + gamma * np.matmul(P, V_old)
        V = np.max(Q, axis=1)
        if np.max(np.abs(V - V_old)) < theta:
            break
    policy = np.argmax(np.round(R + gamma * np.matmul(P, V), decimals=9), axis=1)
    return policy

def load_rl_policy(q_path):
    if not os.path.exists(q_path):
        return None
    q = np.load(q_path)
    return np.argmax(q, axis=1)

def get_state_index(task, comm, local_q, qn1, qn2):
    return int(task * 1815 + comm * 605 + local_q * 121 + qn1 * 11 + qn2)

def generate_facet_grid(policy, name, lambda_val, reward_type, output_dir):
    # Set up matplotlib figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    
    # Custom 4-color map: 0=Blue, 1=Green, 2=Yellow-Green, 3=Red
    colors = ['#3b82f6', '#10b981', '#84cc16', '#ef4444']
    cmap = ListedColormap(colors)
    
    comm_names = ["Poor", "Normal", "Good"]
    task_names = ["Delay-Sensitive (w=2.0)", "Delay-Tolerant (w=0.5)"]
    
    for task in [0, 1]:
        for comm in range(3):
            ax = axes[task, comm]
            
            # Grid of local_q (0..4) vs neighbor_q (0..10, assuming symmetric qn1=qn2)
            grid = np.zeros((5, 11))
            for l_q in range(5):
                for n_q in range(11):
                    s_idx = get_state_index(task, comm, l_q, n_q, n_q)
                    grid[l_q, n_q] = policy[s_idx]
            
            # Plot heatmap using custom categorical colors (origin='lower' to keep local_q=0 at bottom)
            im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='lower', vmin=0, vmax=3)
            
            # Subplot titles and labels
            if task == 0:
                ax.set_title(f"Channel: {comm_names[comm]}", fontsize=12, fontweight='bold')
            if comm == 0:
                ax.set_ylabel(f"{task_names[task]}\nLocal Queue Length", fontsize=11)
            if task == 1:
                ax.set_xlabel("Neighbor Queue (qn1 = qn2)", fontsize=11)
                
            ax.set_yticks(range(5))
            ax.set_xticks(range(0, 11, 2))
            ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
            
    # Add a global legend
    patches = [
        mpatches.Patch(color='#3b82f6', label='Action 0: Local Processing'),
        mpatches.Patch(color='#10b981', label='Action 1: Offload to N1'),
        mpatches.Patch(color='#84cc16', label='Action 2: Offload to N2'),
        mpatches.Patch(color='#ef4444', label='Action 3: Drop Task')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=12)
    
    plt.suptitle(f"Multi-panel Policy Facet Grid: {name} (Lambda={lambda_val}, Reward={reward_type})\nSymmetric Neighbor Queues Slice", 
                 fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    out_path = os.path.join(output_dir, f"policy_facet_grid_{name.lower().replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved facet grid to {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--lambda_val", type=float, default=1.5)
    parser.add_argument("--reward_type", type=str, default="standard")
    args = parser.parse_args()
    
    lambda_val = args.lambda_val
    reward_type = args.reward_type
    res_dir = f"results/L_{lambda_val}_E_{args.episodes}/{reward_type}"
    os.makedirs(res_dir, exist_ok=True)
    
    # 1. DP
    model_path = f"models/mdp_model_{reward_type}_L{lambda_val}.pkl"
    dp_policy = solve_dp(model_path)
    if dp_policy is not None:
        generate_facet_grid(dp_policy, "DP Optimal", lambda_val, reward_type, res_dir)
        
    # 2. Q-Learning
    ql_path = f"results/L_{lambda_val}_E_{args.episodes}/{reward_type}/q_table_ql.npy"
    ql_policy = load_rl_policy(ql_path)
    if ql_policy is not None:
        generate_facet_grid(ql_policy, "Q-Learning", lambda_val, reward_type, res_dir)
        
    # 3. Expected SARSA
    sarsa_path = f"results/L_{lambda_val}_E_{args.episodes}/{reward_type}/q_table_sarsa.npy"
    sarsa_policy = load_rl_policy(sarsa_path)
    if sarsa_policy is not None:
        generate_facet_grid(sarsa_policy, "Expected SARSA", lambda_val, reward_type, res_dir)

if __name__ == "__main__":
    main()
