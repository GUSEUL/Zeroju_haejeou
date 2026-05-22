import os
import pandas as pd

def main():
    lambdas = [0.5, 1.5, 3.5]
    reward_types = ["standard", "sparse", "cliff"]
    
    for r in reward_types:
        print(f"\n=================== REWARD TYPE: {r.upper()} ===================")
        for l in lambdas:
            path = f"results/L_{l}_E_5000/{r}/mdp_final_results.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"\n--- Lambda = {l} ---")
                print(df.to_string(index=False))
            else:
                print(f"Path not found: {path}")

if __name__ == "__main__":
    main()
