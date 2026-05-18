import numpy as np
import pandas as pd
from mdc_gym_env import MDCOffloadingEnv
from compare_baselines import train_sarsa, train_q_learning, evaluate, AllLocalAgent, RandomAgent, ThresholdAgent

def run_stress_test_comparison():
    # 1. Initialize Environments
    env_normal = MDCOffloadingEnv(burst_mode=False)
    env_burst = MDCOffloadingEnv(burst_mode=True)
    
    print("--- Phase 1: Training RL Agents in Normal Conditions ---")
    sarsa_q, _ = train_sarsa(env_normal, episodes=5000)
    ql_q, _ = train_q_learning(env_normal, episodes=5000)
    
    # 2. Define All 5 Agents
    agents = {
        "SARSA": (sarsa_q, True),
        "Q-Learning": (ql_q, True),
        "All-Local": (AllLocalAgent(), False),
        "Random": (RandomAgent(), False),
        "Threshold": (ThresholdAgent(threshold=2), False)
    }
    
    # 3. Evaluation in Burst Mode
    print("\n" + "="*85)
    print(f"{'Agent Name':<15} | {'Burst Avg Reward':<18} | {'Drop Rate %':<15} | {'Energy (J)':<12}")
    print("-" * 85)
    
    stress_results = []
    
    for name, (agent, is_q) in agents.items():
        # Evaluate in the BURST environment
        avg_r, drop_pct, avg_e = evaluate(env_burst, agent, is_q, name)
        print(f"{name:<15} | {avg_r:18.2f} | {drop_pct:14.2f}% | {avg_e:12.2f}")
        stress_results.append({
            "agent": name, 
            "burst_reward": avg_r, 
            "burst_drop_rate_pct": drop_pct,
            "burst_energy": avg_e
        })
    
    print("="*85)
    pd.DataFrame(stress_results).to_csv("stress_test_comparison_results.csv", index=False)
    print("\n[Success] Stress test comparison results saved to 'stress_test_comparison_results.csv'")

if __name__ == "__main__":
    run_stress_test_comparison()
