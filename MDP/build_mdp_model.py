import numpy as np
import os
import pickle
import argparse

def build_model(env_class, lambda_val=1.5):
    env = env_class(arrival_lambda=lambda_val)
    n_states = env.n_states
    n_actions = env.action_space.n
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    print(f"Building MDP Model ({n_states} states) with Lambda={lambda_val}...")
    
    p_task = np.array([0.5, 0.5])
    def get_comm_probs(curr):
        probs = np.zeros(3)
        for change, p in zip([-1, 0, 1], [0.1, 0.8, 0.1]):
            nxt = np.clip(curr + change, 0, 2)
            probs[nxt] += p
        return probs

    for s_idx in range(n_states):
        state = list(np.unravel_index(s_idx, env.observation_space.nvec))
        task, comm, q_l, q_n1, q_n2 = state
        p_comm = get_comm_probs(comm)
        
        for a in range(n_actions):
            samples = 100
            total_r = 0
            queue_transitions = {}
            
            for _ in range(samples):
                env.local_q = q_l
                env.neighbor_qs = np.array([q_n1, q_n2])
                env.comm_state = comm
                env.state = np.array(state)
                
                ns, r, _, _, _ = env.step(a)
                total_r += r
                nq_key = (ns[2], ns[3], ns[4])
                queue_transitions[nq_key] = queue_transitions.get(nq_key, 0) + 1
            
            R[s_idx, a] = total_r / samples
            
            for (nq_l, nq_n1, nq_n2), count in queue_transitions.items():
                p_queue = count / samples
                for nt in range(2):
                    for nc in range(3):
                        ns_idx = env.get_state_index([nt, nc, nq_l, nq_n1, nq_n2])
                        P[s_idx, a, ns_idx] += p_queue * p_task[nt] * p_comm[nc]
                        
        if (s_idx + 1) % 100 == 0: print(f" Progress: {s_idx+1}/{n_states}")
    
    return P, R

if __name__ == "__main__":
    from mdc_mdp_env import MDCMDPEnv
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=1.5, help="Task arrival rate (lambda)")
    args = parser.parse_args()
    
    lambda_val = args.lambda_val
    P, R = build_model(MDCMDPEnv, lambda_val=lambda_val)
    suffix = f"_L{lambda_val}"
    with open(f"mdp_model{suffix}.pkl", "wb") as f: pickle.dump((P, R), f)
    print(f"Model saved to mdp_model{suffix}.pkl")
