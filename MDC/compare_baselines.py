import time
import os
import numpy as np
import pandas as pd
from mdc_gym_env import MDCOffloadingEnv

# --- Utilities for Checkpointing ---
def save_q_table(q_table, filename):
    # Ensure saving in the same directory as the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)
    np.save(path, q_table)
    print(f"  > Checkpoint saved: {path}")

def load_q_table(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        print(f"  > Loading checkpoint: {path}")
        return np.load(path)
    return None

# --- Baseline Agents ---
class AllLocalAgent:
    def choose_action(self, state): return 0

class RandomAgent:
    def choose_action(self, state): return np.random.randint(0, 7)

class ThresholdAgent:
    def __init__(self, threshold=2): self.threshold = threshold
    def choose_action(self, state): return 0 if state[3] < self.threshold else np.random.randint(1, 7)

# --- RL Training Functions ---

def train_sarsa(env, episodes=5000, checkpoint="sarsa_q_table.npy"):
    # Dynamic checkpoint naming based on env.arrival_lambda
    if env.arrival_lambda is not None:
        name_parts = checkpoint.split(".")
        checkpoint = f"{name_parts[0]}_L{env.arrival_lambda}.npy"

    if checkpoint:
        q_table = load_q_table(checkpoint)
        if q_table is not None:
            return q_table, 0.0

    print(f"\n[Training SARSA Agent (Lambda={env.arrival_lambda})...]")
    start_time = time.time()
    q_table = np.zeros((env.n_states, env.action_space.n))
    alpha, gamma = 0.1, 0.95
    epsilon, epsilon_decay, epsilon_min = 1.0, 0.999, 0.01
    logs = []

    for ep in range(episodes):
        s, _ = env.reset()
        si = env.get_state_index(s)
        a = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[si])
        ep_r, ep_d = 0, 0
        
        while True:
            ns, r, term, trunc, info = env.step(a)
            ni = env.get_state_index(ns)
            na = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[ni])
            
            # SARSA Update
            q_table[si, a] += alpha * (r + gamma * q_table[ni, na] - q_table[si, a])
            
            ep_r += r
            if info['is_dropped']: ep_d += 1
            si, a = ni, na
            if term or trunc: break
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        logs.append({"episode": ep, "reward": ep_r, "drops": ep_d})
        if (ep + 1) % 1000 == 0:
            print(f"SARSA Episode {ep+1}/{episodes} | Avg Reward: {ep_r/1000:.2f} | Eps: {epsilon:.3f}")

    end_time = time.time()
    elapsed = end_time - start_time
    # Save log with lambda suffix
    log_name = f"sarsa_train_log_L{env.arrival_lambda}.csv" if env.arrival_lambda else "sarsa_train_log.csv"
    pd.DataFrame(logs).to_csv(log_name, index=False)
    if checkpoint:
        save_q_table(q_table, checkpoint)
    return q_table, elapsed

def train_q_learning(env, episodes=5000, checkpoint="ql_q_table.npy"):
    if env.arrival_lambda is not None:
        name_parts = checkpoint.split(".")
        checkpoint = f"{name_parts[0]}_L{env.arrival_lambda}.npy"

    if checkpoint:
        q_table = load_q_table(checkpoint)
        if q_table is not None:
            return q_table, 0.0

    print(f"\n[Training Q-Learning Agent (Lambda={env.arrival_lambda})...]")
    start_time = time.time()
    q_table = np.zeros((env.n_states, env.action_space.n))
    alpha, gamma = 0.1, 0.95
    epsilon, epsilon_decay, epsilon_min = 1.0, 0.999, 0.01
    logs = []

    for ep in range(episodes):
        s, _ = env.reset()
        si = env.get_state_index(s)
        ep_r, ep_d = 0, 0
        
        while True:
            a = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[si])
            ns, r, term, trunc, info = env.step(a)
            ni = env.get_state_index(ns)
            
            # Q-Learning Update
            q_table[si, a] += alpha * (r + gamma * np.max(q_table[ni]) - q_table[si, a])
            
            ep_r += r
            if info['is_dropped']: ep_d += 1
            si = ni
            if term or trunc: break
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        logs.append({"episode": ep, "reward": ep_r, "drops": ep_d})
        if (ep + 1) % 1000 == 0:
            print(f"Q-Learning Episode {ep+1}/{episodes} | Avg Reward: {ep_r/1000:.2f} | Eps: {epsilon:.3f}")

    end_time = time.time()
    elapsed = end_time - start_time
    log_name = f"q_learning_train_log_L{env.arrival_lambda}.csv" if env.arrival_lambda else "q_learning_train_log.csv"
    pd.DataFrame(logs).to_csv(log_name, index=False)
    if checkpoint:
        save_q_table(q_table, checkpoint)
    return q_table, elapsed

def evaluate(env, agent, is_q=False, name="Agent"):
    rewards, drops, energies = [], [], []
    for _ in range(50):
        s, _ = env.reset()
        ep_r, ep_d, ep_e = 0, 0, 0
        while True:
            a = np.argmax(agent[env.get_state_index(s)]) if is_q else agent.choose_action(s)
            s, r, term, trunc, info = env.step(a)
            ep_r += r
            ep_e += info['energy']
            if info['is_dropped']: ep_d += 1
            if term or trunc: break
        rewards.append(ep_r)
        drops.append(ep_d)
        energies.append(ep_e)
    
    avg_reward = np.mean(rewards)
    drop_rate = (np.mean(drops) / 1000) * 100
    avg_energy = np.mean(energies)
    
    return avg_reward, drop_rate, avg_energy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=None, help="Poisson arrival lambda")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    args = parser.parse_args()

    env = MDCOffloadingEnv(arrival_lambda=args.lambda_val)
    
    sarsa_q, sarsa_time = train_sarsa(env, episodes=args.episodes)
    ql_q, ql_time = train_q_learning(env, episodes=args.episodes)
    
    agents = {
        "SARSA": (sarsa_q, True, sarsa_time),
        "Q-Learning": (ql_q, True, ql_time),
        "All-Local": (AllLocalAgent(), False, 0),
        "Random": (RandomAgent(), False, 0),
        "Threshold": (ThresholdAgent(), False, 0)
    }
    
    print("\n" + "="*85)
    print(f"{'Agent Name':<15} | {'Avg Reward':<12} | {'Drop Rate %':<12} | {'Energy (J)':<12} | {'Time (s)':<8}")
    print("-" * 85)
    
    results = []
    for name, (agent, is_q, t) in agents.items():
        avg_r, drop_pct, avg_e = evaluate(env, agent, is_q, name)
        time_str = f"{t:8.2f}" if t > 0 else "N/A"
        print(f"{name:<15} | {avg_r:12.2f} | {drop_pct:11.2f}% | {avg_e:12.2f} | {time_str}")
        results.append({"agent": name, "reward": avg_r, "drop_rate_pct": drop_pct, "energy": avg_e, "training_time": t})
    
    print("="*85)
    pd.DataFrame(results).to_csv("final_comparison_results.csv", index=False)
    print("\n[Success] Results saved to 'final_comparison_results.csv'")
