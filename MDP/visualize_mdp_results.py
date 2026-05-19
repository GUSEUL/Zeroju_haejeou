import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

def plot_training_comparison(lambda_val=None):
    suffix = f"_L{lambda_val}" if lambda_val else ""
    algos = ["sarsa", "q_learning"]
    plt.figure(figsize=(10, 6))
    for algo in algos:
        path = f"{algo}_log{suffix}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            plt.plot(df["reward"].rolling(window=100).mean(), label=algo.upper())
    plt.title(f"Training Curve (Lambda={lambda_val})")
    plt.xlabel("Episode"); plt.ylabel("Reward (Rolling Avg)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"mdp_training_curves{suffix}.png")

def plot_performance_bars(lambda_val=None):
    suffix = f"_L{lambda_val}" if lambda_val else ""
    path = f"mdp_final_results{suffix}.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(["reward", "drops", "energy"]):
        sns.barplot(data=df, x="agent", y=col, ax=ax[i], palette="muted")
        ax[i].set_title(col.capitalize())
        ax[i].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"mdp_performance_bars{suffix}.png")

def plot_policy_heatmap(policy, env, title, lambda_val=None):
    # Heatmap: CPU_BW (y) vs Local_Q (x)
    heatmap_data = np.zeros((6, 11))
    for cbw in range(6):
        for ql in range(11):
            actions = []
            for task in [0, 1]:
                for chan in [1]:
                    for bnq in [0]:
                        s_idx = env.get_state_index([task, chan, cbw, ql, bnq])
                        a = np.argmax(policy[s_idx]) if policy.ndim > 1 else policy[s_idx]
                        actions.append(a)
            heatmap_data[cbw, ql] = np.bincount(actions).argmax()
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", xticklabels=range(11), yticklabels=range(6))
    plt.title(f"Policy: {title} (0:Loc, 1:N1, 2:N2, 3:Drop)")
    plt.xlabel("Local Queue"); plt.ylabel("CPU_BW State")
    suffix = f"_L{lambda_val}" if lambda_val else ""
    plt.savefig(f"mdp_heatmap_{title.lower().replace(" ", "_")}{suffix}.png")

if __name__ == "__main__":
    from mdc_mdp_env import MDCMDPEnv
    import pickle
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_val", type=float, default=0.5)
    args = p.parse_args()
    env = MDCMDPEnv(arrival_lambda=args.lambda_val)
    plot_training_comparison(args.lambda_val)
    plot_performance_bars(args.lambda_val)
    suffix = f"_L{args.lambda_val}"
    if os.path.exists(f"mdp_model{suffix}.pkl"):
        with open(f"mdp_model{suffix}.pkl", "rb") as f: P, R = pickle.load(f)
        from train_all_mdp import value_iteration
        pol, _, _ = value_iteration(P, R)
        plot_policy_heatmap(pol, env, "DP Optimal", args.lambda_val)

