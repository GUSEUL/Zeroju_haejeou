import pickle
import numpy as np

def check_dp_distribution(model_path):
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
    unique, counts = np.unique(policy, return_counts=True)
    print(f"DP optimal policy action distribution for {model_path}:")
    for a, c in zip(unique, counts):
        print(f"  Action {a}: {c} states ({c/n_states*100:.2f}%)")

if __name__ == "__main__":
    import os
    for f in os.listdir("models"):
        if f.startswith("mdp_model") and f.endswith(".pkl"):
            check_dp_distribution(os.path.join("models", f))
