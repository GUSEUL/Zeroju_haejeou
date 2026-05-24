import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import shutil

# Output directory paths
vis_dir = "visualization"
res_dir = "results"
artifacts_dir = r"C:\Users\sbeen\.gemini\antigravity-cli\brain\a77332d9-2947-4557-b570-286414b9617e"

os.makedirs(vis_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

# Helper to get state index
def get_state_index(task, comm, q_local, q_n1, q_n2):
    return int(task * 1815 + comm * 605 + q_local * 121 + q_n1 * 11 + q_n2)

# Helper to simulate one step
def simulate_step(q_table, state, arrival, task_type, comm_next, local_service_rate=2):
    task, comm, q_l, q_n1, q_n2 = state
    
    # 1. Action lookup from Q-table
    s_idx = get_state_index(task, comm, q_l, q_n1, q_n2)
    action = np.argmax(q_table[s_idx])
    
    # Apply action
    q_l_act = q_l
    q_n1_act = q_n1
    q_n2_act = q_n2
    intentional_drop = (action == 3)
    overflow_drop = False
    
    if action == 0:
        q_l_act += 1
    elif action == 1:
        q_n1_act += 1
    elif action == 2:
        q_n2_act += 1
        
    if q_l_act >= 5:
        overflow_drop = True
        q_l_act = 4
        
    # Process queue tasks
    q_l_served = max(0, q_l_act - local_service_rate)
    
    # Neighbors process tasks stochastically (Option A: 20% chance of 1 task, 80% chance of 0 tasks)
    n1_served_rate = 1 if random.random() < 0.2 else 0
    n2_served_rate = 1 if random.random() < 0.2 else 0
    
    q_n1_served = min(10, max(0, q_n1_act - n1_served_rate))
    q_n2_served = min(10, max(0, q_n2_act - n2_served_rate))
    
    # Background arrivals and overflow
    bg_drops = max(0, q_l_served + arrival - 4)
    q_l_next = min(4, q_l_served + arrival)
    
    next_state = (task_type, comm_next, q_l_next, q_n1_served, q_n2_served)
    
    return next_state, {
        'action': action,
        'q_l_act': q_l_act,
        'q_n1_act': q_n1_act,
        'q_n2_act': q_n2_act,
        'q_l_next': q_l_next,
        'q_n1_next': q_n1_served,
        'q_n2_next': q_n2_served,
        'intentional_drop': intentional_drop,
        'overflow_drop': overflow_drop,
        'bg_drops': bg_drops,
        'arrival': arrival,
        'task_type': task
    }

def generate_animation(lambda_val, ep_std, ep_imp, out_filename):
    print(f" Simulating and generating animation for Lambda = {lambda_val}...")
    
    # Load Q-tables
    q_std_path = f"results/L_{lambda_val}_E_{ep_std}/standard/q_table_ql.csv"
    q_imp_path = f"results/L_{lambda_val}_E_{ep_imp}/improved/q_table_ql.csv"
    
    if not os.path.exists(q_std_path) or not os.path.exists(q_imp_path):
        print(f"Error: Missing CSV files for Lambda={lambda_val}. Make sure conversion completed.")
        return
        
    q_std = np.loadtxt(q_std_path, delimiter=",")
    q_imp = np.loadtxt(q_imp_path, delimiter=",")

    steps = 200
    
    # Pre-generate environmental transitions for consistency
    np.random.seed(42)
    random.seed(42)

    arrivals = np.random.poisson(lambda_val, size=steps)
    task_types = np.random.randint(0, 2, size=steps)

    # Communication state transitions (random walk)
    comm_states = [1]
    for _ in range(steps - 1):
        change = random.choice([-1, 0, 1])
        comm_states.append(int(np.clip(comm_states[-1] + change, 0, 2)))

    # Run simulations
    state_std = (task_types[0], comm_states[0], 0, 0, 0)
    state_imp = (task_types[0], comm_states[0], 0, 0, 0)

    history_std = []
    history_imp = []

    for t in range(steps):
        comm_next = comm_states[t+1] if t < steps - 1 else comm_states[-1]
        task_next = task_types[t+1] if t < steps - 1 else task_types[-1]
        
        state_std, info_std = simulate_step(q_std, state_std, arrivals[t], task_next, comm_next)
        state_imp, info_imp = simulate_step(q_imp, state_imp, arrivals[t], task_next, comm_next)
        
        history_std.append(info_std)
        history_imp.append(info_imp)

    # Setup matplotlib animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    fig.patch.set_facecolor('#0f172a') # Sleek dark background

    # Custom rendering function for a single frame
    def draw_frame(i):
        ax1.clear()
        ax2.clear()
        
        # Sleek dark mode styling
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e293b')
            ax.set_xlim(-1, 12)
            ax.set_ylim(-1, 5)
            ax.axis('off')
            
        info_std = history_std[i]
        info_imp = history_imp[i]
        comm_names = ["Poor", "Normal", "Good"]
        action_names = ["Local", "Offload N1", "Offload N2", "Drop"]
        
        # Helper to draw queue tables
        def draw_queues(ax, info, title):
            # Title
            ax.text(0, 4.4, title, color='#f8fafc', fontsize=14, fontweight='bold')
            
            # Scenario Info
            task_str = "URLLC (Sensitive)" if info['task_type'] == 0 else "eMBB (Tolerant)"
            comm_str = comm_names[comm_states[i]]
            ax.text(0, 4.0, f"Task Arrival: {task_str} | Comm: {comm_str}", color='#94a3b8', fontsize=10)
            ax.text(0, 3.6, f"Incoming Load: +{info['arrival']} tasks | Action Chosen: {action_names[info['action']]}", color='#e2e8f0', fontsize=11, fontweight='semibold')
            
            # Local Queue (Capacity 5)
            ax.text(0, 2.6, "Local Queue (Buffer: 5)", color='#cbd5e1', fontsize=10, fontweight='bold')
            for idx in range(5):
                is_filled = idx < info['q_l_act']
                color = '#3b82f6' if is_filled else '#475569' # Blue for filled, gray for empty
                rect = patches.Rectangle((idx * 1.2, 2.0), 1.0, 0.45, linewidth=1, edgecolor='#64748b', facecolor=color)
                ax.add_patch(rect)
                if is_filled:
                    ax.text(idx * 1.2 + 0.35, 2.12, "T", color='#ffffff', fontsize=9, fontweight='bold')
                    
            # Neighbor 1 Queue (Capacity 10)
            ax.text(0, 1.4, "Neighbor 1 Queue (Buffer: 10)", color='#cbd5e1', fontsize=10, fontweight='bold')
            for idx in range(10):
                is_filled = idx < info['q_n1_act']
                color = '#10b981' if is_filled else '#475569' # Emerald green for N1, gray for empty
                rect = patches.Rectangle((idx * 1.0, 0.8), 0.8, 0.45, linewidth=1, edgecolor='#64748b', facecolor=color)
                ax.add_patch(rect)
                if is_filled:
                    ax.text(idx * 1.0 + 0.25, 0.92, "T", color='#ffffff', fontsize=8, fontweight='semibold')
                    
            # Neighbor 2 Queue (Capacity 10)
            ax.text(0, 0.2, "Neighbor 2 Queue (Buffer: 10)", color='#cbd5e1', fontsize=10, fontweight='bold')
            for idx in range(10):
                is_filled = idx < info['q_n2_act']
                color = '#84cc16' if is_filled else '#475569' # Lime green for N2, gray for empty
                rect = patches.Rectangle((idx * 1.0, -0.4), 0.8, 0.45, linewidth=1, edgecolor='#64748b', facecolor=color)
                ax.add_patch(rect)
                if is_filled:
                    ax.text(idx * 1.0 + 0.25, -0.28, "T", color='#ffffff', fontsize=8, fontweight='semibold')
                    
            # Drop indicators (Red highlighting)
            drop_occurred = info['intentional_drop'] or info['overflow_drop'] or (info['bg_drops'] > 0)
            if drop_occurred:
                drop_type = "Intentional Drop" if info['intentional_drop'] else "Overflow Drop"
                if info['bg_drops'] > 0:
                    drop_type += f" (+{info['bg_drops']} Background)"
                
                # Draw big red banner at the top right of the subplot
                rect = patches.Rectangle((7.5, 3.8), 3.5, 0.5, facecolor='#ef4444', edgecolor='#f87171', alpha=0.9)
                ax.add_patch(rect)
                ax.text(7.7, 3.95, f"DROP: {drop_type}", color='#ffffff', fontsize=8, fontweight='bold')
                
                # Highlight local queue border in red if local drop occurred
                if info['overflow_drop'] or info['bg_drops'] > 0:
                    rect_border = patches.Rectangle((-0.1, 1.9), 6.2, 0.65, linewidth=2, edgecolor='#ef4444', facecolor='none')
                    ax.add_patch(rect_border)
                    
        # Draw both subplots
        draw_queues(ax1, info_std, "Standard Reward Policy (Baseline)")
        draw_queues(ax2, info_imp, "QoS-Aware Improved Reward Policy")
        
        # Global title showing progress
        fig.suptitle(f"MDP Queue Offloading Simulation (Step {i+1}/{steps})  |  Arrival Rate Lambda = {lambda_val}", color='#cbd5e1', fontsize=16, fontweight='bold', y=0.95)

    # Build and save animation
    anim = animation.FuncAnimation(fig, draw_frame, frames=steps, interval=600)
    
    gif_vis_path = os.path.join(vis_dir, out_filename)
    gif_res_path = os.path.join(res_dir, out_filename)
    gif_art_path = os.path.join(artifacts_dir, out_filename)
    
    anim.save(gif_vis_path, writer='pillow', fps=1.5)
    shutil.copyfile(gif_vis_path, gif_res_path)
    shutil.copyfile(gif_vis_path, gif_art_path)
    
    plt.close()
    print(f" Finished saving {out_filename} to visualization, results, and artifacts.")

if __name__ == "__main__":
    # Generate GIF for Lambda = 1.5
    generate_animation(lambda_val=1.5, ep_std=30000, ep_imp=20000, out_filename="queue_simulation_L1.5.gif")
    
    # Generate GIF for Lambda = 3.0
    generate_animation(lambda_val=3.0, ep_std=30000, ep_imp=20000, out_filename="queue_simulation_L3.0.gif")
    
    print("All GIFs generated successfully.")
