import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_training_metrics(lambda_val=None):
    """
    Plots Reward and Drop Count over episodes for SARSA and Q-Learning.
    """
    suffix = f"_L{lambda_val}" if lambda_val is not None else ""
    sarsa_log = f"sarsa_train_log{suffix}.csv"
    ql_log = f"q_learning_train_log{suffix}.csv"
    
    # Check if logs exist
    logs = []
    if os.path.exists(sarsa_log):
        logs.append(("SARSA", sarsa_log, "blue"))
    if os.path.exists(ql_log):
        logs.append(("Q-Learning", ql_log, "green"))
        
    if not logs:
        print(f"No log files found for Lambda={lambda_val}. Please train the agents first.")
        return

    # Create subplots: Top for Reward, Bottom for Drops
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    window = 100 # Rolling average window

    for name, path, color in logs:
        df = pd.read_csv(path)
        
        # 1. Plot Reward
        reward_smooth = df['reward'].rolling(window=window).mean()
        ax1.plot(reward_smooth, label=f'{name} Reward', color=color, linewidth=2)
        
        # 2. Plot Drops
        drops_smooth = df['drops'].rolling(window=window).mean()
        ax2.plot(drops_smooth, label=f'{name} Drops', color=color, linestyle='--', alpha=0.8)

    # Styling Reward Plot
    ax1.set_title(f'Training Metrics Over Time (Lambda={lambda_val if lambda_val else "Default"})', fontsize=15)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Styling Drops Plot
    ax2.set_ylabel('Total Drops', fontsize=12)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    out_file = f'training_metrics_detail{suffix}.png'
    plt.savefig(out_file)
    print(f"Successfully generated detailed metrics plot: {out_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=None, help="Lambda value used during training")
    args = parser.parse_args()
    
    plot_training_metrics(args.lambda_val)
