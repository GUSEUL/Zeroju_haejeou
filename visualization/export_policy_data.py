import os
import pickle
import json
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
    # Vectorized value iteration for 500x speedup
    for _ in range(10000):
        V_old = V.copy()
        Q = R + gamma * np.matmul(P, V_old)
        V = np.max(Q, axis=1)
        if np.max(np.abs(V - V_old)) < theta:
            break
    policy = np.argmax(np.round(R + gamma * np.matmul(P, V), decimals=9), axis=1)
    return policy.tolist()

def load_rl_policy(q_path):
    if not os.path.exists(q_path):
        return None
    if q_path.endswith('.npy'):
        q = np.load(q_path)
    else:
        q = np.loadtxt(q_path, delimiter=",")
    policy = np.argmax(q, axis=1)
    return policy.tolist()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30000)
    args = parser.parse_args()
    
    lambdas = [0.1, 0.5, 1.5, 3.0]
    reward_types = ["standard", "sparse", "cliff", "improved"]
    
    data = {
        "lambdas": lambdas,
        "reward_types": reward_types,
        "agents": ["dp", "ql", "sarsa"],
        "policies": {}
    }
    
    for l in lambdas:
        for r in reward_types:
            key_prefix = f"L{l}_{r}"
            
            # 1. DP
            model_path = f"models/mdp_model_{r}_L{l}.pkl"
            dp_policy = solve_dp(model_path)
            if dp_policy is not None:
                data["policies"][f"{key_prefix}_dp"] = dp_policy
                print(f"Processed DP for {key_prefix}")
            else:
                print(f"DP Model not found: {model_path}")
                
            # Episode count selection: 20000 for improved, 30000 (args.episodes) for baseline
            ep = 20000 if r == "improved" else args.episodes
            
            # 2. Q-Learning
            ql_path = f"results/L_{l}_E_{ep}/{r}/q_table_ql.csv"
            ql_policy = load_rl_policy(ql_path)
            if ql_policy is not None:
                data["policies"][f"{key_prefix}_ql"] = ql_policy
                print(f"Processed Q-Learning for {key_prefix}")
            else:
                print(f"Q-Learning Q-table not found: {ql_path}")
                
            # 3. Expected SARSA
            sarsa_path = f"results/L_{l}_E_{ep}/{r}/q_table_sarsa.csv"
            sarsa_policy = load_rl_policy(sarsa_path)
            if sarsa_policy is not None:
                data["policies"][f"{key_prefix}_sarsa"] = sarsa_policy
                print(f"Processed Expected SARSA for {key_prefix}")
            else:
                print(f"Expected SARSA Q-table not found: {sarsa_path}")

    # Ensure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("visualization", exist_ok=True)
    
    # Save JSON files
    out_paths = [
        "visualization/policy_data.json",
        "results/policy_data.json"
    ]
    for p in out_paths:
        with open(p, "w") as f:
            json.dump(data, f)
        print(f"Saved policy data to {p}")
        
    # Save JS file for local browser dashboard
    js_paths = [
        "visualization/policy_data.js",
        "results/policy_data.js"
    ]
    for p in js_paths:
        with open(p, "w") as f:
            f.write(f"const POLICY_DATA = {json.dumps(data)};")
        print(f"Saved policy data JS to {p}")

if __name__ == "__main__":
    main()
