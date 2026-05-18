import numpy as np
from mdc_gym_env import MDCOffloadingEnv
from compare_baselines import train_sarsa, evaluate

def stress_test():
    # 1. Normal Environment
    env_normal = MDCOffloadingEnv(burst_mode=False)
    # 2. Burst Environment (Higher arrival rate)
    env_burst = MDCOffloadingEnv(burst_mode=True)
    
    print("Training Agent in Normal Env...")
    q_table = train_sarsa(env_normal, episodes=2000)
    
    print("\n--- Stress Test (SARSA Performance) ---")
    r_n, d_n = evaluate(env_normal, q_table, is_q=True)
    print(f"[Normal Mode] Avg Reward: {r_n:8.2f} | Avg Drops: {d_n:5.1f}")
    
    r_b, d_b = evaluate(env_burst, q_table, is_q=True)
    print(f"[Burst  Mode] Avg Reward: {r_b:8.2f} | Avg Drops: {d_b:5.1f}")
    
    print("\nConclusion: Check if the agent uses 'Intentional Drop' or 'Heavy Offloading' to survive the burst.")

if __name__ == "__main__":
    stress_test()
