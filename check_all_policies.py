import os
import pickle
import numpy as np

def solve_dp(model_path):
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        P, R = pickle.load(f)
    n_states = P.shape[0]
    V = np.zeros(n_states)
    gamma = 0.95
    theta = 1e-12
    for _ in range(10000):
        V_old = V.copy()
        for s in range(n_states):
            V[s] = np.max(R[s] + gamma * np.dot(P[s], V_old))
        if np.max(np.abs(V - V_old)) < theta:
            break
    policy = np.array([np.argmax(np.round(R[s] + gamma * np.dot(P[s], V), decimals=9)) for s in range(n_states)])
    return policy

def analyze_q_table(q_path):
    if not os.path.exists(q_path):
        return None, None
    if q_path.endswith('.npy'):
        q = np.load(q_path)
    else:
        q = np.loadtxt(q_path, delimiter=",")
    # Visited mask: any Q-value not equal to initialization value -150.0
    visited_mask = np.any(q != -150.0, axis=1)
    num_visited = np.sum(visited_mask)
    pct_visited = num_visited / len(q) * 100.0
    
    # Calculate policy across all states
    policy = np.argmax(q, axis=1)
    return pct_visited, policy

def print_distribution(policy, name):
    if policy is None:
        return "N/A"
    unique, counts = np.unique(policy, return_counts=True)
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for a, c in zip(unique, counts):
        dist[a] = c
    total = len(policy)
    return f"{name}: Loc={dist[0]} ({dist[0]/total*100:.1f}%), N1={dist[1]} ({dist[1]/total*100:.1f}%), N2={dist[2]} ({dist[2]/total*100:.1f}%), Drop={dist[3]} ({dist[3]/total*100:.1f}%)"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()
    
    lambdas = [0.5, 1.5, 3.5]
    reward_types = ["standard", "sparse", "cliff"]
    
    print("=" * 100)
    print(f"Results for Episodes = {args.episodes}")
    print(f"{'Scenario':<25} | {'Agent':<15} | {'Visited %':<10} | {'Action Distribution':<50}")
    print("=" * 100)
    
    for l in lambdas:
        for r in reward_types:
            scenario_name = f"L={l}, R={r}"
            
            # 1. DP
            model_path = f"models/mdp_model_{r}_L{l}.pkl"
            dp_policy = solve_dp(model_path)
            if dp_policy is not None:
                print(f"{scenario_name:<25} | {'DP Optimal':<15} | {'100.0%':<10} | {print_distribution(dp_policy, 'DP')}")
            
            # 2. Q-Learning
            ql_path = f"results/L_{l}_E_{args.episodes}/{r}/q_table_ql.csv"
            ql_visited, ql_policy = analyze_q_table(ql_path)
            if ql_policy is not None:
                print(f"{'':<25} | {'Q-Learning':<15} | {f'{ql_visited:.2f}%':<10} | {print_distribution(ql_policy, 'QL')}")
                
            # 3. Expected SARSA
            sarsa_path = f"results/L_{l}_E_{args.episodes}/{r}/q_table_sarsa.csv"
            sarsa_visited, sarsa_policy = analyze_q_table(sarsa_path)
            if sarsa_policy is not None:
                print(f"{'':<25} | {'Expected SARSA':<15} | {f'{sarsa_visited:.2f}%':<10} | {print_distribution(sarsa_policy, 'SARSA')}")
            print("-" * 100)

if __name__ == "__main__":
    main()
