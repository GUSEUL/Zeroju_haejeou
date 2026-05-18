import numpy as np
import pandas as pd
import argparse
import os
from mdc_gym_env import MDCOffloadingEnv

def save_state_values(lambda_val=None, agent_type="sarsa"):
    """
    Calculates V(s) = max_a Q(s,a) for all states and saves to CSV.
    """
    env = MDCOffloadingEnv(arrival_lambda=lambda_val)
    
    # Construct filename
    prefix = "sarsa_q_table" if agent_type.lower() == "sarsa" else "ql_q_table"
    suffix = f"_L{lambda_val}" if lambda_val is not None else ""
    checkpoint = f"{prefix}{suffix}.npy"
    
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint {checkpoint} not found. Please train the agent first.")
        return

    # Load Q-table
    q_table = np.load(checkpoint)
    print(f"Loaded Q-table from {checkpoint}")

    state_values = []
    
    # Iterate through all possible states in the observation space
    # MultiDiscrete([2, 3, 3, 4, 2]) -> Task, Channel, CPU, Queue, BW
    for task in range(2):
        for channel in range(3):
            for cpu in range(3):
                for queue in range(4):
                    for bw in range(2):
                        state = np.array([task, channel, cpu, queue, bw])
                        idx = env.get_state_index(state)
                        
                        # V(s) is the maximum Q-value for that state
                        v_s = np.max(q_table[idx])
                        best_action = np.argmax(q_table[idx])
                        
                        state_values.append({
                            "Task": "URLLC" if task == 0 else "eMBB",
                            "Channel": ["Poor", "Normal", "Good"][channel],
                            "CPU": ["Low", "Normal", "Congested"][cpu],
                            "Queue": ["Empty", "Smooth", "Warning", "Critical"][queue],
                            "BW": "Insufficient" if bw == 0 else "Sufficient",
                            "State_Value": v_s,
                            "Best_Action": best_action
                        })

    # Save to CSV
    df = pd.DataFrame(state_values)
    out_file = f"state_values_{agent_type.lower()}{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"Successfully saved state values to {out_file}")
    
    # Display top 5 highest value states
    print("\nTop 5 High-Value States (Optimal Conditions):")
    print(df.sort_values(by="State_Value", ascending=False).head(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=None, help="Lambda value used during training")
    parser.add_argument("--agent", type=str, default="sarsa", choices=["sarsa", "ql"], help="Agent type (sarsa or ql)")
    args = parser.parse_args()
    
    save_state_values(args.lambda_val, args.agent)
