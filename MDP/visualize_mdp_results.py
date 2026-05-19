import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

def plot_training_comparison(lambda_val=None, reward_type="standard", res_dir="."):
    algos = ["sarsa", "q-learning"]
    plt.figure(figsize=(10, 6))
    found = False
    for algo in algos:
        path = os.path.join(res_dir, f"{algo}_log.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                plt.plot(df["reward"].rolling(window=100).mean(), label=algo.upper())
                found = True
    if found:
        plt.title(f"Training Curve (Reward={reward_type}, Lambda={lambda_val})")
        plt.xlabel("Episode"); plt.ylabel("Reward (Rolling Avg)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(res_dir, f"mdp_training_curves.png"))
    plt.close()

def plot_performance_bars(lambda_val=None, reward_type="standard", res_dir="."):
    path = os.path.join(res_dir, "mdp_final_results.csv")
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    # Exclude Random agent for clearer comparison
    df = df[df["agent"] != "Random"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(["reward", "drops", "energy"]):
        sns.barplot(data=df, x="agent", y=col, ax=ax[i], palette="muted", hue="agent", legend=False)
        ax[i].set_title(f"{col.capitalize()} ({reward_type})")
        ax[i].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, f"mdp_performance_bars.png"))
    plt.close()

def plot_policy_heatmap(policy, env, title, lambda_val=None, reward_type="standard", res_dir="."):
    heatmap_data = np.zeros((3, 5))
    for comm in range(3):
        for ql in range(5):
            actions = []
            for task in [0, 1]:
                for qn1 in [0, 5]: 
                    for qn2 in [0, 5]:
                        s_idx = env.get_state_index([task, comm, ql, qn1, qn2])
                        if policy.ndim > 1:
                            a = np.argmax(policy[s_idx])
                        else:
                            a = policy[s_idx]
                        actions.append(a)
            heatmap_data[comm, ql] = np.bincount(actions).argmax()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", xticklabels=range(5), yticklabels=["Poor", "Normal", "Good"])
    plt.title(f"Policy: {title} ({reward_type}) (0:Loc, 1:N1, 2:N2, 3:Drop)")
    plt.xlabel("Local Queue"); plt.ylabel("Comm State")
    plt.savefig(os.path.join(res_dir, f"mdp_heatmap_{title.lower().replace(' ', '_')}.png"))
    plt.close()

if __name__ == "__main__":
    from mdc_mdp_env import MDCMDPEnv
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=1.5)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--reward_type", type=str, default="standard", choices=["standard", "sparse", "cliff"])
    args = parser.parse_args()
    
    res_dir = f"results/L_{args.lambda_val}_E_{args.episodes}/{args.reward_type}"
    env = MDCMDPEnv(arrival_lambda=args.lambda_val, reward_type=args.reward_type)
    
    plot_training_comparison(args.lambda_val, args.reward_type, res_dir)
    plot_performance_bars(args.lambda_val, args.reward_type, res_dir)
    
    suffix = f"_{args.reward_type}_L{args.lambda_val}"
    model_path = os.path.join("models", f"mdp_model{suffix}.pkl")
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f: P, R = pickle.load(f)
        from train_all_mdp import value_iteration
        pol, _, _ = value_iteration(P, R)
        plot_policy_heatmap(pol, env, "DP Optimal", args.lambda_val, args.reward_type, res_dir)
        
    for algo in [("sarsa", "SARSA"), ("ql", "Q-Learning")]:
        q_path = os.path.join(res_dir, f"q_table_{algo[0]}.npy")
        if os.path.exists(q_path):
            q_table = np.load(q_path)
            plot_policy_heatmap(q_table, env, algo[1], args.lambda_val, args.reward_type, res_dir)
