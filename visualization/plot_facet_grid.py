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
    if q_path.endswith('.npy'):
        q = np.load(q_path)
    else:
        q = np.loadtxt(q_path, delimiter=",")
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
            
            # Extract 2D slice for local_q vs qn1 (fixing qn2=0 for visualization)
            grid = np.zeros((5, 11))
            for local_q in range(5):
                for qn1 in range(11):
                    s_idx = get_state_index(task, comm, local_q, qn1, 0)
                    grid[local_q, qn1] = policy[s_idx]
            
            # Plot slice
            im = ax.imshow(grid, cmap=cmap, origin='lower', aspect='auto', vmin=0, vmax=3)
            
            # Titles and labels
            if task == 0:
                ax.set_title(f"Channel: {comm_names[comm]}")
            if comm == 0:
                ax.set_ylabel(f"{task_names[task]}\nLocal Queue Size")
            if task == 1:
                ax.set_xlabel("Neighbor 1 Queue Size")
                
    # Add title and legend
    fig.suptitle(f"Optimal Policy Map Slice (Neighbor 2 Queue = 0) - {name}\nLambda={lambda_val}, Reward={reward_type}", fontsize=16)
    
    # Create legend patches
    patch_loc = mpatches.Patch(color='#3b82f6', label='Local')
    patch_n1 = mpatches.Patch(color='#10b981', label='Neighbor 1')
    patch_n2 = mpatches.Patch(color='#84cc16', label='Neighbor 2')
    patch_drop = mpatches.Patch(color='#ef4444', label='Drop')
    
    fig.legend(handles=[patch_loc, patch_n1, patch_n2, patch_drop], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    
    # Save plot
    file_name = f"policy_grid_{name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved facet grid to {save_path}")

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
    ql_path = f"results/L_{lambda_val}_E_{args.episodes}/{reward_type}/q_table_ql.csv"
    ql_policy = load_rl_policy(ql_path)
    if ql_policy is not None:
        generate_facet_grid(ql_policy, "Q-Learning", lambda_val, reward_type, res_dir)
        
    # 3. Expected SARSA
    sarsa_path = f"results/L_{lambda_val}_E_{args.episodes}/{reward_type}/q_table_sarsa.csv"
    sarsa_policy = load_rl_policy(sarsa_path)
    if sarsa_policy is not None:
        generate_facet_grid(sarsa_policy, "Expected SARSA", lambda_val, reward_type, res_dir)

if __name__ == "__main__":
    main()
