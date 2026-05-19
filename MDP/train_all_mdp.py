import numpy as np
import pandas as pd
import os
import pickle
import time
from mdc_mdp_env import MDCMDPEnv

def value_iteration(P, R, gamma=0.95, theta=1e-6):
    n_states = P.shape[0]
    V = np.zeros(n_states)
    start_time = time.time()
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            V[s] = np.max(R[s] + gamma * np.dot(P[s], V))
            delta = max(delta, abs(v - V[s]))
        if delta < theta: break
    policy = np.array([np.argmax(R[s] + gamma * np.dot(P[s], V)) for s in range(n_states)])
    return policy, V, time.time() - start_time

def policy_iteration(P, R, gamma=0.95, theta=1e-6):
    n_states = P.shape[0]
    policy = np.zeros(n_states, dtype=int)
    start_time = time.time()
    while True:
        V = np.zeros(n_states)
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]; a = policy[s]
                V[s] = R[s, a] + gamma * np.dot(P[s, a], V)
                delta = max(delta, abs(v - V[s]))
            if delta < theta: break
        stable = True
        for s in range(n_states):
            old_a = policy[s]
            policy[s] = np.argmax(R[s] + gamma * np.dot(P[s], V))
            if old_a != policy[s]: stable = False
        if stable: break
    return policy, V, time.time() - start_time

def train_q_learning(env, episodes=5000, gamma=0.95, suffix=""):
    q_path = f"q_table_ql{suffix}.npy"
    if os.path.exists(q_path):
        print(f" Loading checkpoint: {q_path}")
        return np.load(q_path), [], 0.0
    
    q = np.zeros((env.n_states, env.action_space.n))
    alpha = 0.1; eps = 1.0; eps_min = 0.01; decay = np.exp(np.log(eps_min)/(episodes*0.8))
    logs = []
    start_time = time.time()
    for ep in range(episodes):
        s, _ = env.reset(); ep_r = 0
        while True:
            si = env.get_state_index(s)
            a = env.action_space.sample() if np.random.rand() < eps else np.argmax(q[si])
            ns, r, term, trunc, _ = env.step(a)
            ni = env.get_state_index(ns)
            q[si, a] += alpha * (r + gamma * np.max(q[ni]) - q[si, a])
            ep_r += r; s = ns
            if term or trunc: break
        if (ep + 1) % 500 == 0: print(f" Episode {ep+1}/{episodes}...")
        eps = max(eps_min, eps * decay)
        logs.append({"episode": ep, "reward": ep_r})
    
    np.save(q_path, q)
    return q, logs, time.time() - start_time

def train_sarsa(env, episodes=5000, gamma=0.95, suffix=""):
    q_path = f"q_table_sarsa{suffix}.npy"
    if os.path.exists(q_path):
        print(f" Loading checkpoint: {q_path}")
        return np.load(q_path), [], 0.0

    q = np.zeros((env.n_states, env.action_space.n))
    alpha = 0.1; eps = 1.0; eps_min = 0.01; decay = np.exp(np.log(eps_min)/(episodes*0.8))
    logs = []
    start_time = time.time()
    for ep in range(episodes):
        s, _ = env.reset(); si = env.get_state_index(s)
        a = env.action_space.sample() if np.random.rand() < eps else np.argmax(q[si])
        ep_r = 0
        while True:
            ns, r, term, trunc, _ = env.step(a)
            ni = env.get_state_index(ns)
            na = env.action_space.sample() if np.random.rand() < eps else np.argmax(q[ni])
            q[si, a] += alpha * (r + gamma * q[ni, na] - q[si, a])
            ep_r += r; si = ni; a = na
            if term or trunc: break
        if (ep + 1) % 500 == 0: print(f" Episode {ep+1}/{episodes}...")
        eps = max(eps_min, eps * decay)
        logs.append({"episode": ep, "reward": ep_r})
        
    np.save(q_path, q)
    return q, logs, time.time() - start_time

def evaluate(env, pol, is_q=False):
    rr, dd, ee = [], [], []
    for _ in range(50):
        s, _ = env.reset(); er, ed, ee_episode = 0, 0, 0
        while True:
            a = np.argmax(pol[env.get_state_index(s)]) if is_q else pol[env.get_state_index(s)]
            s, r, term, trunc, info = env.step(a)
            er += r
            ee_episode += info.get("energy", 0)
            if info.get("is_dropped", False): ed += 1
            if term or trunc: break
        rr.append(er); dd.append(ed); ee.append(ee_episode)
    return np.mean(rr), np.mean(dd), np.mean(ee)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_val", type=float, default=1.5)
    p.add_argument("--episodes", type=int, default=5000)
    args = p.parse_args()
    env = MDCMDPEnv(arrival_lambda=args.lambda_val)
    results = []
    suffix = f"_L{args.lambda_val}"
    if os.path.exists(f"mdp_model{suffix}.pkl"):
        with open(f"mdp_model{suffix}.pkl", "rb") as f: P, R = pickle.load(f)
        for n, f in [("Policy Iteration", policy_iteration), ("Value Iteration", value_iteration)]:
            pol, _, t = f(P, R); r, d, e = evaluate(env, pol)
            results.append({"agent": n, "reward": r, "drops": d, "energy": e, "time": t})
    for n, f in [("SARSA", train_sarsa), ("Q-Learning", train_q_learning)]:
        q, logs, t = f(env, episodes=args.episodes, suffix=suffix)
        if len(logs) > 0:
            pd.DataFrame(logs).to_csv(f"{n.lower()}_log{suffix}.csv", index=False)
        r, d, e = evaluate(env, q, is_q=True)
        results.append({"agent": n, "reward": r, "drops": d, "energy": e, "time": t})
    r, d, e = evaluate(env, np.zeros(env.n_states, dtype=int))
    results.append({"agent": "All-Local", "reward": r, "drops": d, "energy": e, "time": 0})
    r, d, e = evaluate(env, np.random.randint(0, 4, size=env.n_states))
    results.append({"agent": "Random", "reward": r, "drops": d, "energy": e, "time": 0})
    pd.DataFrame(results).to_csv(f"mdp_final_results{suffix}.csv", index=False)
    print("Done.")
