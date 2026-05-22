import pickle
import numpy as np
import os

def value_iteration(P, R, gamma=0.95, theta=1e-12):
    n_states = P.shape[0]
    V = np.zeros(n_states)
    it = 0
    while True:
        delta = 0
        V_old = V.copy()
        # Jacobi style (strictly standard)
        for s in range(n_states):
            V[s] = np.max(R[s] + gamma * np.dot(P[s], V_old))
            delta = max(delta, abs(V_old[s] - V[s]))
        it += 1
        if delta < theta: break
    
    # Consistent tie-breaking by rounding
    policy = np.array([np.argmax(np.round(R[s] + gamma * np.dot(P[s], V), decimals=9)) for s in range(n_states)])
    return policy, V, it

def policy_iteration(P, R, gamma=0.95, theta=1e-12):
    n_states = P.shape[0]
    policy = np.zeros(n_states, dtype=int)
    it = 0
    while True:
        V = np.zeros(n_states)
        # Policy Evaluation
        while True:
            delta = 0
            V_old = V.copy()
            for s in range(n_states):
                a = policy[s]
                V[s] = R[s, a] + gamma * np.dot(P[s, a], V_old)
                delta = max(delta, abs(V_old[s] - V[s]))
            if delta < theta: break
            
        # Policy Improvement
        stable = True
        for s in range(n_states):
            old_a = policy[s]
            # Consistent tie-breaking by rounding
            policy[s] = np.argmax(np.round(R[s] + gamma * np.dot(P[s], V), decimals=9))
            if old_a != policy[s]: stable = False
        it += 1
        if stable: break
    return policy, V, it

# Load standard L1.5 model
model_path = "models/mdp_model_standard_L1.5.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        P, R = pickle.load(f)
        
    pol_vi, V_vi, it_vi = value_iteration(P, R)
    pol_pi, V_pi, it_pi = policy_iteration(P, R)
    
    diff = np.sum(pol_vi != pol_pi)
    v_diff = np.max(np.abs(V_vi - V_pi))
    print(f"Mismatch count between VI and PI policies: {diff} / {len(pol_vi)}")
    print(f"Max absolute difference in state values: {v_diff}")
    print(f"VI Iterations: {it_vi}, PI Iterations: {it_pi}")
else:
    print("Model file not found!")
