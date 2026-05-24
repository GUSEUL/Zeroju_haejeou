import os
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=100000)
args = parser.parse_args()

lambdas = [0.5, 1.0, 1.5]
reward_types = ["standard", "cliff"]
base_dir = "results"

for l in lambdas:
    print(f"\n### Task Arrival Rate Lambda = {l} (Comparative Summary)")
    print("| Reward Formulation | Agent / Algorithm | Expected Reward | Avg Pending / Episode | Avg Energy / Episode | Episodes | solver/train Time |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for r in reward_types:
        ep = args.episodes
        csv_path = os.path.join(base_dir, f"L_{l}_E_{ep}", r, "mdp_final_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Reorder rows: PI, VI, SARSA, Q-Learning, Random
            df['agent_order'] = df['agent'].map({
                'Policy Iteration': 0,
                'Value Iteration': 1,
                'SARSA': 2,
                'Q-Learning': 3,
                'Random': 4
            })
            df = df.sort_values('agent_order')
            
            first_row = True
            for _, row in df.iterrows():
                rf_col = f"**{r.capitalize()}**" if first_row else ""
                first_row = False
                
                reward_val = f"{row['reward']:.2f}"
                pending_val = f"{row['pending']:.2f}"
                energy_val = f"{row['energy']:.2f}"
                time_val = f"{row['time']:.2f}s" if row['time'] > 0 else "N/A"
                
                print(f"| {rf_col} | {row['agent']} | {reward_val} | {pending_val} | {energy_val} | {ep} | {time_val} |")
        else:
            print(f"| **{r.capitalize()}** | *No results file found* | | | | | |")
