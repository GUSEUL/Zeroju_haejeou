import numpy as np
import os
import pickle
import argparse
from scipy.stats import poisson

def build_model(env_class, lambda_val=1.5, reward_type="standard"):
    n_states = 2 * 3 * 5 * 11 * 11
    n_actions = 4
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    print(f"Building Analytical MDP Model ({n_states} states) with Lambda={lambda_val}, Reward={reward_type}...")
    
    service_rates = [2, 2, 2]
    channel_factors = [1.5, 1.0, 0.5]
    energy_costs = [0.8, 0.5, 0.3]
    
    w = 0.6
    beta = 5.0
    beta_neighbor = 5.0
    gamma = 5.0
    
    max_delay = 5.0
    max_energy = 1.0
    max_local_q = 5.0
    
    def get_comm_probs(curr):
        probs = np.zeros(3)
        for change, p in zip([-1, 0, 1], [0.1, 0.8, 0.1]):
            nxt = np.clip(curr + change, 0, 2)
            probs[nxt] += p
        return probs

    # Transition probability for a single neighbor queue
    # trans_n[q_after_action, q_next]
    # neighbor queue after action can be up to 11
    trans_n = np.zeros((12, 11))
    for q_curr in range(12):
        for s_n, prob in zip([0, 1], [0.8, 0.2]):
            q_served = max(0, q_curr - s_n)
            q_next = min(10, q_served)
            trans_n[q_curr, q_next] += prob
            
    # Transition probability for local queue
    # trans_l[q_after_action, comm, q_next]
    trans_l = np.zeros((5, 3, 5))
    for q_curr in range(5):
        for comm in range(3):
            s_rate = service_rates[comm]
            q_served = max(0, q_curr - s_rate)
            for arr in range(100):
                q_next = min(4, q_served + arr)
                p_a = poisson.pmf(arr, lambda_val)
                trans_l[q_curr, comm, q_next] += p_a

    # Loop over all states
    for s_idx in range(n_states):
        temp = s_idx
        q_n2 = temp % 11
        temp //= 11
        q_n1 = temp % 11
        temp //= 11
        q_l = temp % 5
        temp //= 5
        comm = temp % 3
        task = temp // 3
        
        p_comm = get_comm_probs(comm)
        
        for a in range(n_actions):
            q_l_act = q_l
            q_n1_act = q_n1
            q_n2_act = q_n2
            is_dropped = False
            
            delay_trans, delay_comp = 0.0, 0.0
            energy_consumed = 0.0
            
            if a == 0: # Local
                s_rate = service_rates[comm]
                delay_comp = (q_l + 1) / (s_rate * 2.0)
                energy_consumed = energy_costs[comm]
                q_l_act += 1
            elif a == 1: # Offload to N1
                chan_factor = channel_factors[comm]
                delay_trans = chan_factor * 1.0
                energy_consumed = energy_costs[comm] * 0.6
                delay_comp = (q_n1 + 1) / 8.0 + 0.05
                q_n1_act += 1
            elif a == 2: # Offload to N2
                chan_factor = channel_factors[comm]
                delay_trans = chan_factor * 1.0
                energy_consumed = energy_costs[comm] * 0.6
                delay_comp = (q_n2 + 1) / 8.0 + 0.05
                q_n2_act += 1
            elif a == 3: # Drop
                is_dropped = True
                
            if q_l_act >= 5:
                is_dropped = True
                q_l_act = 4
                
            # Expected background drops calculation
            q_served = max(0, q_l_act - service_rates[comm])
            exp_bg_drops = 0.0
            for arr in range(100):
                p_a = poisson.pmf(arr, lambda_val)
                if q_served + arr > 4:
                    exp_bg_drops += p_a * (q_served + arr - 4)

            # Reward
            total_delay = delay_trans + delay_comp
            if reward_type == "improved":
                w_task = 2.5 if task == 0 else 0.5
            else:
                w_task = 2.0 if task == 0 else 0.5
            
            norm_delay = np.clip(total_delay / max_delay, 0.0, 1.0)
            norm_energy = np.clip(energy_consumed / max_energy, 0.0, 1.0)
            cost_de = w_task * w * norm_delay + (1.0 - w) * norm_energy
            
            penalty_neighbor = 0.0
            if reward_type == "improved":
                beta_neighbor_task = 6.0 if task == 0 else 3.0
                if a == 1:
                    penalty_neighbor = beta_neighbor_task * ((q_n1_act / 10.0) ** 2)
                elif a == 2:
                    penalty_neighbor = beta_neighbor_task * ((q_n2_act / 10.0) ** 2)
            else:
                if a == 1:
                    penalty_neighbor = beta_neighbor * ((q_n1_act / 10.0) ** 2)
                elif a == 2:
                    penalty_neighbor = beta_neighbor * ((q_n2_act / 10.0) ** 2)

            if reward_type == "sparse":
                # Sparse with mild regularization
                reward = -100.0 if is_dropped else (-0.1 - 0.01 * cost_de)
                reward -= 100.0 * exp_bg_drops
            elif reward_type == "cliff":
                penalty_drop = 1000.0 if is_dropped else 0.0
                reward = -(cost_de + penalty_drop + penalty_neighbor + 1000.0 * exp_bg_drops)
            elif reward_type == "improved":
                gamma_task = 30.0 if task == 0 else 10.0
                beta_task = 8.0 if task == 0 else 3.0
                penalty_queue = beta_task * ((q_l_act / max_local_q) ** 2)
                penalty_drop = gamma_task if is_dropped else 0.0
                reward = -(cost_de + penalty_queue + penalty_drop + penalty_neighbor + gamma_task * exp_bg_drops)
            else: # standard
                penalty_queue = beta * ((q_l_act / max_local_q) ** 2)
                penalty_drop = gamma if is_dropped else 0.0
                reward = - (cost_de + penalty_queue + penalty_drop + penalty_neighbor + gamma * exp_bg_drops)
                
            R[s_idx, a] = reward
            
            p_nq_l = trans_l[q_l_act, comm]
            p_nq_n1 = trans_n[q_n1_act]
            p_nq_n2 = trans_n[q_n2_act]
            
            non_zero_nq_l = np.flatnonzero(p_nq_l)
            non_zero_nq_n1 = np.flatnonzero(p_nq_n1)
            non_zero_nq_n2 = np.flatnonzero(p_nq_n2)
            
            for nq_l in non_zero_nq_l:
                pl = p_nq_l[nq_l]
                for nq_n1 in non_zero_nq_n1:
                    pl1 = pl * p_nq_n1[nq_n1]
                    for nq_n2 in non_zero_nq_n2:
                        pl2 = pl1 * p_nq_n2[nq_n2]
                        
                        for nt in range(2):
                            p_t = pl2 * 0.5
                            for nc in range(3):
                                p_final = p_t * p_comm[nc]
                                ns_idx = nt * 1815 + nc * 605 + nq_l * 121 + nq_n1 * 11 + nq_n2
                                P[s_idx, a, ns_idx] += p_final
                                
        if (s_idx + 1) % 500 == 0:
            print(f" Progress: {s_idx+1}/{n_states}")
            
    return P, R

if __name__ == "__main__":
    from mdc_mdp_env import MDCMDPEnv
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=1.5)
    parser.add_argument("--reward_type", type=str, default="standard")
    args = parser.parse_args()
    
    P, R = build_model(MDCMDPEnv, lambda_val=args.lambda_val, reward_type=args.reward_type)
    
    os.makedirs("models", exist_ok=True)
    suffix = f"_{args.reward_type}_L{args.lambda_val}"
    model_path = os.path.join("models", f"mdp_model{suffix}.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump((P, R), f)
    print(f"Model saved to {model_path}")
