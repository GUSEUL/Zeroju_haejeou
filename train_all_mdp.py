import numpy as np
import pandas as pd
import os
import pickle
import time
import argparse
from mdc_mdp_env import MDCMDPEnv

# Offsets for seed scheduling so train / eval / random-policy streams don't overlap.
EVAL_SEED_OFFSET = 10000
RANDOM_POLICY_SEED_OFFSET = 99991

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

def train_q_learning(env, episodes=5000, gamma=0.95, output_dir=".", base_seed=42):
    q_path = os.path.join(output_dir, "q_table_ql.npy")
    if os.path.exists(q_path):
        print(f" Loading checkpoint: {q_path}")
        return np.load(q_path), [], 0.0

    q = np.zeros((env.n_states, env.action_space.n))
    alpha = 0.1; eps = 1.0; eps_min = 0.01; decay = np.exp(np.log(eps_min)/(episodes*0.8))
    logs = []
    # Dedicated RNG for ε-greedy / action_space.sample seeding (independent of env stream).
    rng = np.random.default_rng(base_seed)
    env.action_space.seed(int(base_seed))
    start_time = time.time()
    for ep in range(episodes):
        s, _ = env.reset(seed=base_seed + ep); ep_r = 0
        while True:
            si = env.get_state_index(s)
            a = env.action_space.sample() if rng.random() < eps else int(np.argmax(q[si]))
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

def train_sarsa(env, episodes=5000, gamma=0.95, output_dir=".", base_seed=42):
    q_path = os.path.join(output_dir, "q_table_sarsa.npy")
    if os.path.exists(q_path):
        print(f" Loading checkpoint: {q_path}")
        return np.load(q_path), [], 0.0

    q = np.zeros((env.n_states, env.action_space.n))
    alpha = 0.1; eps = 1.0; eps_min = 0.01; decay = np.exp(np.log(eps_min)/(episodes*0.8))
    logs = []
    rng = np.random.default_rng(base_seed + 1)  # +1 to decouple from QL stream
    env.action_space.seed(int(base_seed) + 1)
    start_time = time.time()
    for ep in range(episodes):
        s, _ = env.reset(seed=base_seed + ep); si = env.get_state_index(s)
        a = env.action_space.sample() if rng.random() < eps else int(np.argmax(q[si]))
        ep_r = 0
        while True:
            ns, r, term, trunc, _ = env.step(a)
            ni = env.get_state_index(ns)
            na = env.action_space.sample() if rng.random() < eps else int(np.argmax(q[ni]))
            q[si, a] += alpha * (r + gamma * q[ni, na] - q[si, a])
            ep_r += r; si = ni; a = na
            if term or trunc: break
        if (ep + 1) % 500 == 0: print(f" Episode {ep+1}/{episodes}...")
        eps = max(eps_min, eps * decay)
        logs.append({"episode": ep, "reward": ep_r})

    np.save(q_path, q)
    return q, logs, time.time() - start_time

def evaluate(env, pol, is_q=False, n_episodes=500, base_seed=42):
    """Evaluate a policy and return mean per-episode KPIs.

    Returns
    -------
    dict with keys: reward, agent_drops, env_drops, admitted, completed, energy
    """
    rr, ad, ed_env, aa, cc, ee = [], [], [], [], [], []
    eval_seed = base_seed + EVAL_SEED_OFFSET
    for ep in range(n_episodes):
        s, _ = env.reset(seed=eval_seed + ep)
        er = 0.0
        agent_drops = 0
        env_drops = 0
        admitted = 0
        completed = 0
        energy_total = 0.0
        while True:
            a = int(np.argmax(pol[env.get_state_index(s)])) if is_q else int(pol[env.get_state_index(s)])
            s, r, term, trunc, info = env.step(a)
            er += r
            energy_total += info.get("energy", 0.0)
            agent_drops += int(info.get("agent_drop", 1 if info.get("is_dropped", False) else 0))
            env_drops += int(info.get("env_drop", 0))
            admitted += int(info.get("is_admitted", False))
            completed += int(info.get("completed_count", 0))
            if term or trunc: break
        rr.append(er); ad.append(agent_drops); ed_env.append(env_drops)
        aa.append(admitted); cc.append(completed); ee.append(energy_total)
    return {
        "reward": float(np.mean(rr)),
        "agent_drops": float(np.mean(ad)),
        "env_drops": float(np.mean(ed_env)),
        "admitted": float(np.mean(aa)),
        "completed": float(np.mean(cc)),
        "energy": float(np.mean(ee)),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=1.5)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--reward_type", type=str, default="standard")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for env/agent reproducibility")
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
        for n, fn in [("Policy Iteration", policy_iteration), ("Value Iteration", value_iteration)]:
            pol, _, t = fn(P, R)
            m = evaluate(env, pol, base_seed=args.seed)
            results.append({"agent": n, **m, "time": t})

    for n, fn in [("SARSA", train_sarsa), ("Q-Learning", train_q_learning)]:
        q, logs, t = fn(env, episodes=args.episodes, output_dir=res_dir, base_seed=args.seed)
        if len(logs) > 0:
            pd.DataFrame(logs).to_csv(os.path.join(res_dir, f"{n.lower()}_log.csv"), index=False)
        m = evaluate(env, q, is_q=True, base_seed=args.seed)
        results.append({"agent": n, **m, "time": t})

    # Random baseline policy — seed the action sampler for reproducibility.
    rng_rand = np.random.default_rng(args.seed + RANDOM_POLICY_SEED_OFFSET)
    rand_pol = rng_rand.integers(0, 4, size=env.n_states)
    m = evaluate(env, rand_pol, base_seed=args.seed)
    results.append({"agent": "Random", **m, "time": 0})

    final_res_path = os.path.join(res_dir, "mdp_final_results.csv")
    pd.DataFrame(results).to_csv(final_res_path, index=False)
    print(f"Results saved to {final_res_path}")
    print("Done.")
