import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
    
    # 1. Reconstruct all 3,630 states
    states = []
    for s in range(3630):
        task = s // 1815
        rem = s % 1815
        comm = rem // 605
        rem = rem % 605
        local_q = rem // 121
        rem = rem % 121
        qn1 = rem // 11
        qn2 = rem % 11
        states.append([task, comm, local_q, qn1, qn2])
    states = np.array(states, dtype=float)
    
    # 2. Scale features for t-SNE
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)
    
    # 3. Fit t-SNE
    print("Running t-SNE on 3,630 states...")
    tsne = TSNE(n_components=2, perplexity=40, n_iter_without_progress=1000, random_state=42, n_jobs=-1)
    states_2d = tsne.fit_transform(states_scaled)
    print("t-SNE embedding computed.")
    
    # Custom 4-color map: 0=Blue, 1=Green, 2=Yellow-Green, 3=Red
    colors = ['#3b82f6', '#10b981', '#84cc16', '#ef4444']
    cmap = ListedColormap(colors)
    
    # Policies to plot
    policies = {}
    
    # Load DP
    model_path = os.path.join("models", f"mdp_model_{reward_type}_L{lambda_val}.pkl")
    print(f"Attempting to load DP model from: {model_path}")
    dp_pol = solve_dp(model_path)
    if dp_pol is not None:
        policies["DP Optimal"] = dp_pol
        print("Successfully loaded DP Optimal policy.")
    else:
        print(f"Warning: DP model file not found: {model_path}")
        
    # Load Q-Learning
    ql_path = f"{res_dir}/q_table_ql.npy"
    ql_pol = load_rl_policy(ql_path)
    if ql_pol is not None:
        policies["Q-Learning"] = ql_pol
        print("Successfully loaded Q-Learning policy.")
        
    # Load Expected SARSA
    sarsa_path = f"{res_dir}/q_table_sarsa.npy"
    sarsa_pol = load_rl_policy(sarsa_path)
    if sarsa_pol is not None:
        policies["Expected SARSA"] = sarsa_pol
        print("Successfully loaded Expected SARSA policy.")
        
    if not policies:
        print("Error: No policies found to plot. Please check if model files or RL results exist for the given parameters.")
        return

    # Generate scatter plot for each policy
    for name, policy in policies.items():
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1], c=policy, cmap=cmap, s=8, alpha=0.7)
        
        # Grid/labels
        plt.title(f"t-SNE State Space Embedding: {name}\n(Lambda={lambda_val}, Reward={reward_type})", 
                  fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Component 1", fontsize=11)
        plt.ylabel("t-SNE Component 2", fontsize=11)
        
        # Legend
        patches = [
            mpatches.Patch(color='#3b82f6', label='Action 0: Local Processing'),
            mpatches.Patch(color='#10b981', label='Action 1: Offload to N1'),
            mpatches.Patch(color='#84cc16', label='Action 2: Offload to N2'),
            mpatches.Patch(color='#ef4444', label='Action 3: Drop Task')
        ]
        plt.legend(handles=patches, loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(res_dir, f"policy_tsne_{name.lower().replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved t-SNE plot to {out_path}")
        
        # print("Displaying plot...")
        # plt.show()
        plt.close()

if __name__ == "__main__":
    main()
