import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import shutil
import pickle

# Output directory paths
vis_dir = "visualization"
res_dir = "results"
artifacts_dir = r"C:\Users\sbeen\OneDrive\Desktop\RL_project - 복사본" # Save inside workspace root
appdata_artifacts_dir = r"C:\Users\sbeen\.gemini\antigravity-cli\brain\a77332d9-2947-4557-b570-286414b9617e"

os.makedirs(vis_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
os.makedirs(appdata_artifacts_dir, exist_ok=True)

# Helper to get state index
def get_state_index(task, comm, q_local, q_n1, q_n2):
    return int(task * 1815 + comm * 605 + q_local * 121 + q_n1 * 11 + q_n2)

# Vectorized Value Iteration for DP Optimal Policy calculation
def solve_dp_optimal(P, R, gamma=0.95, theta=1e-12):
    n_states = P.shape[0]
    V = np.zeros(n_states)
    print("Solving DP Optimal Policy via Value Iteration...")
    while True:
        # Vectorized Bellman Backup
        V_new = np.max(R + gamma * np.dot(P.reshape(n_states * 4, n_states), V).reshape(n_states, 4), axis=1)
        if np.max(np.abs(V_new - V)) < theta:
            V = V_new
            break
        V = V_new
    policy = np.argmax(R + gamma * np.dot(P.reshape(n_states * 4, n_states), V).reshape(n_states, 4), axis=1)
    print("DP Optimal Policy solved successfully.")
    return policy

# Helper to simulate one step with infinite pending buffer queue dynamics
def simulate_step(policy_or_q, state, pending_buffer, arrival, task_type, comm_next, n1_served_rate, n2_served_rate, local_service_rate=2):
    task, comm, q_l, q_n1, q_n2 = state
    
    # Action lookup (policy is 1D array of actions, Q-table is 2D array of action values)
    s_idx = get_state_index(task, comm, q_l, q_n1, q_n2)
    if policy_or_q.ndim == 1:
        action = int(policy_or_q[s_idx])
    else:
        action = int(np.argmax(policy_or_q[s_idx]))
    
    # Apply action
    q_l_act = q_l
    q_n1_act = q_n1
    q_n2_act = q_n2
    intentional_pending = (action == 3)
    overflow_pending = False
    
    if action == 0:
        q_l_act += 1
    elif action == 1:
        q_n1_act += 1
    elif action == 2:
        q_n2_act += 1
    elif action == 3: # Pending action: put in the pending buffer
        pending_buffer += 1
        
    if q_l_act >= 5:
        overflow_pending = True
        overflow_count = q_l_act - 4
        pending_buffer += overflow_count
        q_l_act = 4
        
    if q_n1_act >= 11:
        overflow_pending = True
        overflow_count = q_n1_act - 10
        pending_buffer += overflow_count
        q_n1_act = 10
        
    if q_n2_act >= 11:
        overflow_pending = True
        overflow_count = q_n2_act - 10
        pending_buffer += overflow_count
        q_n2_act = 10
        
    # Process local queue tasks
    q_l_served = max(0, q_l_act - local_service_rate)
    
    # Process neighbor queue tasks stochastically (clipped to 10)
    q_n1_served = min(10, max(0, q_n1_act - n1_served_rate))
    q_n2_served = min(10, max(0, q_n2_act - n2_served_rate))
    
    # Background arrivals and local queue overflow
    num_bg_pending = max(0, q_l_served + arrival - 4)
    q_l_next = min(4, q_l_served + arrival)
    pending_buffer += num_bg_pending
    
    # Replenish local queue from the infinite pending buffer if space becomes available
    if q_l_next < 4 and pending_buffer > 0:
        to_move = min(4 - q_l_next, pending_buffer)
        q_l_next += to_move
        pending_buffer -= to_move
    
    next_state = (task_type, comm_next, q_l_next, q_n1_served, q_n2_served)
    
    return next_state, pending_buffer, {
        'action': action,
        'q_l_act': q_l_act,
        'q_n1_act': q_n1_act,
        'q_n2_act': q_n2_act,
        'q_l_next': q_l_next,
        'q_n1_next': q_n1_served,
        'q_n2_next': q_n2_served,
        'intentional_pending': intentional_pending,
        'overflow_pending': overflow_pending,
        'bg_pending': num_bg_pending,
        'arrival': arrival,
        'task_type': task,
        'pending_buffer_size': pending_buffer
    }

def generate_comparison_animation(lambda_val, episodes, reward_type, out_filename):
    print(f"\n==================================================")
    print(f"Generating Algorithm Comparison for Lambda={lambda_val}, Reward={reward_type}")
    print(f"==================================================")
    
    # Load Q-tables
    q_ql_path = f"results/L_{lambda_val}_E_{episodes}/{reward_type}/q_table_ql.csv"
    q_sarsa_path = f"results/L_{lambda_val}_E_{episodes}/{reward_type}/q_table_sarsa.csv"
    model_path = f"models/mdp_model_{reward_type}_L{lambda_val}.pkl"
    
    if not os.path.exists(q_ql_path) or not os.path.exists(q_sarsa_path) or not os.path.exists(model_path):
        print(f"Error: Missing files for Lambda={lambda_val}, Reward={reward_type}.")
        print(f"Check availability of:\n - {q_ql_path}\n - {q_sarsa_path}\n - {model_path}")
        return False
        
    q_ql = np.loadtxt(q_ql_path, delimiter=",")
    q_sarsa = np.loadtxt(q_sarsa_path, delimiter=",")
    
    with open(model_path, "rb") as f:
        P, R = pickle.load(f)
        
    # Solve for DP Optimal Policy
    policy_dp = solve_dp_optimal(P, R)

    steps = 200
    
    # Pre-generate environmental transitions for strict consistency
    np.random.seed(42)
    random.seed(42)

    arrivals = np.random.poisson(lambda_val, size=steps)
    task_types = np.random.randint(0, 2, size=steps)

    # Communication state transitions (random walk)
    comm_states = [1]
    for _ in range(steps - 1):
        change = random.choice([-1, 0, 1])
        comm_states.append(int(np.clip(comm_states[-1] + change, 0, 2)))

    # Neighbor processing rates (Option A: 20% chance of 1 task, 80% chance of 0 tasks)
    n1_served_rates = [1 if random.random() < 0.2 else 0 for _ in range(steps)]
    n2_served_rates = [1 if random.random() < 0.2 else 0 for _ in range(steps)]

    # Run simulations for all 3 policies
    state_dp = (task_types[0], comm_states[0], 0, 0, 0)
    state_sarsa = (task_types[0], comm_states[0], 0, 0, 0)
    state_ql = (task_types[0], comm_states[0], 0, 0, 0)

    pending_buffer_dp = 0
    pending_buffer_sarsa = 0
    pending_buffer_ql = 0

    history_dp = []
    history_sarsa = []
    history_ql = []

    for t in range(steps):
        comm_next = comm_states[t+1] if t < steps - 1 else comm_states[-1]
        task_next = task_types[t+1] if t < steps - 1 else task_types[-1]
        
        state_dp, pending_buffer_dp, info_dp = simulate_step(
            policy_dp, state_dp, pending_buffer_dp, arrivals[t], task_next, comm_next,
            n1_served_rates[t], n2_served_rates[t]
        )
        state_sarsa, pending_buffer_sarsa, info_sarsa = simulate_step(
            q_sarsa, state_sarsa, pending_buffer_sarsa, arrivals[t], task_next, comm_next,
            n1_served_rates[t], n2_served_rates[t]
        )
        state_ql, pending_buffer_ql, info_ql = simulate_step(
            q_ql, state_ql, pending_buffer_ql, arrivals[t], task_next, comm_next,
            n1_served_rates[t], n2_served_rates[t]
        )
        
        history_dp.append(info_dp)
        history_sarsa.append(info_sarsa)
        history_ql.append(info_ql)

    # Setup matplotlib 3-column animation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8.5))
    fig.patch.set_facecolor('#0f172a') # Sleek dark background

    # Custom rendering function for a single frame
    def draw_frame(i):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Sleek dark mode styling
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#1e293b')
            ax.set_xlim(-1, 12)
            ax.set_ylim(-1, 5)
            ax.axis('off')
            
        info_dp = history_dp[i]
        info_sarsa = history_sarsa[i]
        info_ql = history_ql[i]
        
        comm_names = ["Poor", "Normal", "Good"]
        action_names = ["Local Process", "Offload N1", "Offload N2", "Pending"]
        
        # Helper to draw queue tables for a single subplot
        def draw_queues(ax, info, title):
            # Title
            ax.text(0, 4.4, title, color='#f8fafc', fontsize=14, fontweight='bold')
            
            # Scenario Info
            task_str = "URLLC (Sensitive)" if info['task_type'] == 0 else "eMBB (Tolerant)"
            comm_str = comm_names[comm_states[i]]
            ax.text(0, 4.0, f"Task Type: {task_str} | Comm Channel: {comm_str}", color='#94a3b8', fontsize=10)
            ax.text(0, 3.6, f"Arrivals: +{info['arrival']} tasks | Chosen Action: {action_names[info['action']]}", color='#e2e8f0', fontsize=11, fontweight='semibold')
            
            # Colors for queues: shades of blue for tasks (as requested), slate for empty slots
            local_filled_color = '#3b82f6'  # Medium blue
            n1_filled_color = '#60a5fa'     # Light blue
            n2_filled_color = '#1d4ed8'     # Darker blue
            empty_color = '#475569'         # Slate gray for empty
            
            # Local Queue (Capacity 5)
            ax.text(0, 2.6, f"Local Queue (Capacity: 5) | Pending Buffer: {info['pending_buffer_size']}", color='#cbd5e1', fontsize=10, fontweight='bold')
            for idx in range(5):
                is_filled = idx < info['q_l_act']
                color = local_filled_color if is_filled else empty_color
                rect = patches.Rectangle((idx * 1.2, 2.0), 1.0, 0.45, linewidth=1, edgecolor='#64748b', facecolor=color)
                ax.add_patch(rect)
                if is_filled:
                    ax.text(idx * 1.2 + 0.35, 2.12, "T", color='#ffffff', fontsize=9, fontweight='bold')
                    
            # Neighbor 1 Queue (Capacity 10)
            ax.text(0, 1.4, "Neighbor 1 Queue (Capacity: 10) | Pending Buffer: 0", color='#cbd5e1', fontsize=10, fontweight='bold')
            for idx in range(10):
                is_filled = idx < info['q_n1_act']
                color = n1_filled_color if is_filled else empty_color
                rect = patches.Rectangle((idx * 1.0, 0.8), 0.8, 0.45, linewidth=1, edgecolor='#64748b', facecolor=color)
                ax.add_patch(rect)
                if is_filled:
                    ax.text(idx * 1.0 + 0.25, 0.92, "T", color='#ffffff', fontsize=8, fontweight='semibold')
                    
            # Neighbor 2 Queue (Capacity 10)
            ax.text(0, 0.2, "Neighbor 2 Queue (Capacity: 10) | Pending Buffer: 0", color='#cbd5e1', fontsize=10, fontweight='bold')
            for idx in range(10):
                is_filled = idx < info['q_n2_act']
                color = n2_filled_color if is_filled else empty_color
                rect = patches.Rectangle((idx * 1.0, -0.4), 0.8, 0.45, linewidth=1, edgecolor='#64748b', facecolor=color)
                ax.add_patch(rect)
                if is_filled:
                    ax.text(idx * 1.0 + 0.25, -0.28, "T", color='#ffffff', fontsize=8, fontweight='semibold')
                    
            # Pending indicators (Red highlighting for pending events)
            pending_occurred = info['intentional_pending'] or info['overflow_pending'] or (info['bg_pending'] > 0)
            if pending_occurred:
                pending_type = "Intentional Pend" if info['intentional_pending'] else "Overflow Pend"
                if info['bg_pending'] > 0:
                    pending_type += f" (+{info['bg_pending']} BG)"
                
                # Draw red banner at the top right
                rect = patches.Rectangle((7.5, 3.8), 3.5, 0.5, facecolor='#ef4444', edgecolor='#f87171', alpha=0.95)
                ax.add_patch(rect)
                ax.text(7.6, 3.95, f"PEND: {pending_type}", color='#ffffff', fontsize=7.5, fontweight='bold')
                
                # Highlight local queue border in red if local pending occurred
                if info['overflow_pending'] or info['bg_pending'] > 0:
                    rect_border = patches.Rectangle((-0.1, 1.9), 6.2, 0.65, linewidth=2, edgecolor='#ef4444', facecolor='none')
                    ax.add_patch(rect_border)
                    
        # Draw all three subplots
        draw_queues(ax1, info_dp, "DP Optimal Policy")
        draw_queues(ax2, info_sarsa, "Expected SARSA Policy")
        draw_queues(ax3, info_ql, "Q-Learning Policy")
        
        # Global title showing progress
        reward_title = "Cliff Penalty Reward" if reward_type == "cliff" else "QoS-Aware Standard Reward"
        fig.suptitle(
            f"MDP Queue Offloading: Policy Comparison (Step {i+1}/{steps})\n"
            f"Lambda = {lambda_val} | Reward Model: {reward_title}",
            color='#f1f5f9', fontsize=16, fontweight='bold', y=0.96
        )

    # Build and save animation
    anim = animation.FuncAnimation(fig, draw_frame, frames=steps, interval=600)
    
    gif_vis_path = os.path.join(vis_dir, out_filename)
    gif_res_path = os.path.join(res_dir, out_filename)
    gif_art_path = os.path.join(artifacts_dir, out_filename)
    gif_appdata_path = os.path.join(appdata_artifacts_dir, out_filename)
    
    # Save the file using pillow writer
    print(f"Saving animation to {gif_vis_path}...")
    anim.save(gif_vis_path, writer='pillow', fps=1.5)
    
    # Replicate file to output directories
    shutil.copyfile(gif_vis_path, gif_res_path)
    shutil.copyfile(gif_vis_path, gif_art_path)
    shutil.copyfile(gif_vis_path, gif_appdata_path)
    
    plt.close()
    print(f"Successfully saved {out_filename} to multiple directories.")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    args = parser.parse_args()
    for lv in [0.5, 1.0, 1.5]:
        generate_comparison_animation(
            lambda_val=lv, 
            episodes=args.episodes, 
            reward_type="standard", 
            out_filename=f"policy_comparison_L{lv}_standard.gif"
        )
        
        generate_comparison_animation(
            lambda_val=lv, 
            episodes=args.episodes, 
            reward_type="cliff", 
            out_filename=f"policy_comparison_L{lv}_cliff.gif"
        )
    
    print("\nAll policy comparison GIFs generated successfully.")
