import numpy as np
import pandas as pd
from unstable_mdc_env import UnstableMDCEnv
from compare_baselines import train_sarsa, train_q_learning, evaluate

def run_unstable_comparison():
    env = UnstableMDCEnv()
    
    print("\n[Phase 1] Training Agents in UNSTABLE Environment...")
    # Train agents on the unstable environment so they learn the volatility
    sarsa_q, _ = train_sarsa(env, episodes=5000, checkpoint="sarsa_q_table_unstable.npy")
    ql_q, _ = train_q_learning(env, episodes=5000, checkpoint="ql_q_table_unstable.npy")
    
    agents = {
        "SARSA (Conservative)": (sarsa_q, True),
        "Q-Learning (Aggressive)": (ql_q, True)
    }
    
    print("\n[Phase 2] Evaluating Agents in UNSTABLE Environment...")
    print("\n" + "="*65)
    print(f"{'Agent Name':<25} | {'Avg Reward':<12} | {'Drop Rate (%)':<12} | {'Energy (J)':<12}")
    print("-" * 75)
    
    results = []
    for name, (agent, is_q) in agents.items():
        # Evaluate using the same unstable environment
        avg_r, drop_pct, avg_e = evaluate(env, agent, is_q, name)
        print(f"{name:<25} | {avg_r:12.2f} | {drop_pct:11.2f}% | {avg_e:12.2f}")
        results.append({
            "agent": name, 
            "reward": avg_r, 
            "drop_rate_pct": drop_pct,
            "energy": avg_e
        })
    
    print("="*75)
    
    pd.DataFrame(results).to_csv("unstable_network_comparison.csv", index=False)
    print("\n[Conclusion]")
    print("In this highly stochastic environment, check if SARSA's risk-averse")
    print("strategy results in a lower drop rate and better average reward")
    print("compared to Q-Learning's optimistic but risky offloading strategy.")

if __name__ == "__main__":
    run_unstable_comparison()
