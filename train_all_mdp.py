import numpy as np
import pandas as pd
import os
import pickle
import time
import argparse
import random
from mdc_mdp_env import MDCMDPEnv

def value_iteration(P, R, gamma=0.95, theta=1e-12):
    n_states = P.shape[0]
    V = np.zeros(n_states)
    start_time = time.time()
    while True:
        delta = 0
        V_old = V.copy()
        for s in range(n_states):
            V[s] = np.max(R[s] + gamma * np.dot(P[s], V_old))
            delta = max(delta, abs(V_old[s] - V[s]))
        if delta < theta: break
    policy = np.array([np.argmax(np.round(R[s] + gamma * np.dot(P[s], V), decimals=9)) for s in range(n_states)])
    return policy, V, time.time() - start_time

def policy_iteration(P, R, gamma=0.95, theta=1e-12):
    n_states = P.shape[0]
    policy = np.zeros(n_states, dtype=int)
    start_time = time.time()
    while True:
        V = np.zeros(n_states)
        while True:
            delta = 0
            V_old = V.copy()
            for s in range(n_states):
                a = policy[s]
                V[s] = R[s, a] + gamma * np.dot(P[s, a], V_old)
                delta = max(delta, abs(V_old[s] - V[s]))
            if delta < theta: break
        stable = True
        for s in range(n_states):
            old_a = policy[s]
            action_values = R[s] + gamma * np.dot(P[s], V)
            best_a = np.argmax(action_values)
            if action_values[best_a] > action_values[old_a] + 1e-11:
                policy[s] = best_a
                stable = False
        if stable: break
    return policy, V, time.time() - start_time

def train_q_learning(env, episodes=5000, gamma=0.95, output_dir="."):
    q_path = os.path.join(output_dir, "q_table_ql.csv")
    if os.path.exists(q_path):
        print(f" Loading checkpoint: {q_path}")
        return np.loadtxt(q_path, delimiter=","), [], 0.0
    
    q = np.full((env.n_states, env.action_space.n), -150.0)
    alpha = 0.1; eps = 1.0; eps_min = 0.01; decay = np.exp(np.log(eps_min)/(episodes*0.6))
    logs = []
    start_time = time.time()
    for ep in range(episodes):
        s, _ = env.reset(options={"random_start": True}); ep_r = 0
        t = 0
        while True:
            si = env.get_state_index(s)
            a = env.action_space.sample() if np.random.rand() < eps else np.argmax(q[si])
            ns, r, term, trunc, _ = env.step(a)
            ni = env.get_state_index(ns)
            q[si, a] += alpha * (r + gamma * np.max(q[ni]) - q[si, a])
            ep_r += (gamma ** t) * r; s = ns
            t += 1
            if term or trunc: break
        if (ep + 1) % 500 == 0: print(f" Episode {ep+1}/{episodes}...")
        eps = max(eps_min, eps * decay)
        logs.append({"episode": ep, "reward": ep_r})
    
    np.savetxt(q_path, q, delimiter=",")
    return q, logs, time.time() - start_time

def train_sarsa(env, episodes=5000, gamma=0.95, output_dir="."):
    q_path = os.path.join(output_dir, "q_table_sarsa.csv")
    if os.path.exists(q_path):
        print(f" Loading checkpoint: {q_path}")
        return np.loadtxt(q_path, delimiter=","), [], 0.0

    q = np.full((env.n_states, env.action_space.n), -150.0)
    alpha = 0.1; eps = 1.0; eps_min = 0.01; decay = np.exp(np.log(eps_min)/(episodes*0.6))
    logs = []
    start_time = time.time()
    for ep in range(episodes):
        s, _ = env.reset(options={"random_start": True}); si = env.get_state_index(s)
        ep_r = 0
        t = 0
        while True:
            # Epsilon-greedy action selection
            a = env.action_space.sample() if np.random.rand() < eps else np.argmax(q[si])
            ns, r, term, trunc, _ = env.step(a)
            ni = env.get_state_index(ns)
            
            # Expected SARSA update
            best_a = np.argmax(q[ni])
            expected_v = (1.0 - eps) * q[ni, best_a] + (eps / 4.0) * np.sum(q[ni])
            
            q[si, a] += alpha * (r + gamma * expected_v - q[si, a])
            ep_r += (gamma ** t) * r; si = ni
            t += 1
            if term or trunc: break
        if (ep + 1) % 500 == 0: print(f" Episode {ep+1}/{episodes}...")
        eps = max(eps_min, eps * decay)
        logs.append({"episode": ep, "reward": ep_r})
        
    np.savetxt(q_path, q, delimiter=",")
    return q, logs, time.time() - start_time

def evaluate(env, pol, is_q=False, gamma=0.95):
    np.random.seed(42)
    random.seed(42)
    rr, dd, ee = [], [], []
    for _ in range(500):
        s, _ = env.reset(); er, ed, ee_episode = 0, 0, 0
        t = 0
        while True:
            a = np.argmax(pol[env.get_state_index(s)]) if is_q else pol[env.get_state_index(s)]
            s, r, term, trunc, info = env.step(a)
            er += (gamma ** t) * r
            ee_episode += info.get("energy", 0)
            ed += info.get("pending_count", info.get("dropped_count", 1 if info.get("is_pending", info.get("is_dropped", False)) else 0))
            t += 1
            if term or trunc: break
        rr.append(er); dd.append(ed); ee.append(ee_episode)
    return np.mean(rr), np.mean(dd), np.mean(ee)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=1.5)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--reward_type", type=str, default="standard")
    args = parser.parse_args()
    
    # 결과 폴더 생성
    res_dir = f"results/L_{args.lambda_val}_E_{args.episodes}/{args.reward_type}"
    os.makedirs(res_dir, exist_ok=True)
    
    env = MDCMDPEnv(arrival_lambda=args.lambda_val, reward_type=args.reward_type)
    results = []
    
    # 모델 로드 (models 폴더에서 찾기)
    suffix = f"_{args.reward_type}_L{args.lambda_val}"
    model_path = os.path.join("models", f"mdp_model{suffix}.pkl")
    
    if os.path.exists(model_path):
        print(f"Loading MDP model: {model_path}")
        with open(model_path, "rb") as f: P, R = pickle.load(f)
        for n, f in [("Policy Iteration", policy_iteration), ("Value Iteration", value_iteration)]:
            pol, _, t = f(P, R)
            r, d, e = evaluate(env, pol)
            results.append({"agent": n, "reward": r, "pending": d, "energy": e, "time": t})
            
    for n, f in [("SARSA", train_sarsa), ("Q-Learning", train_q_learning)]:
        q, logs, t = f(env, episodes=args.episodes, output_dir=res_dir)
        if len(logs) > 0:
            pd.DataFrame(logs).to_csv(os.path.join(res_dir, f"{n.lower()}_log.csv"), index=False)
        r, d, e = evaluate(env, q, is_q=True)
        results.append({"agent": n, "reward": r, "pending": d, "energy": e, "time": t})
        
    r, d, e = evaluate(env, np.random.randint(0, 4, size=env.n_states))
    results.append({"agent": "Random", "reward": r, "pending": d, "energy": e, "time": 0})
    
    final_res_path = os.path.join(res_dir, "mdp_final_results.csv")
    pd.DataFrame(results).to_csv(final_res_path, index=False)
    print(f"Results saved to {final_res_path}")
    print("Done.")
