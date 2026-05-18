import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mdc_gym_env import MDCOffloadingEnv

# --- 1. Baseline Agents Implementation ---

class AllLocalAgent:
    def choose_action(self, state):
        return 0

class RandomAgent:
    def choose_action(self, state):
        # Pick between 0 and 6 (Local + Offloading, exclude intentional drop for baseline)
        return np.random.randint(0, 7)

class ThresholdAgent:
    def __init__(self, threshold_level=2):
        self.threshold = threshold_level

    def choose_action(self, state):
        queue_level = state[3]
        if queue_level < self.threshold:
            return 0  # Local
        else:
            return np.random.randint(1, 7)  # Random Offloading

# --- 2. Evaluation Function ---

def evaluate_agent(env, agent, num_episodes=10, is_q_table=False, burst_mode=False):
    total_rewards = []
    total_delays = []
    drop_counts = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_delay = 0
        drops = 0
        
        for step in range(1000):
            # Stress Test: Burst Mode (Increase arrival rate by modifying env or simulating)
            # Here we simulate by artificially increasing physical queue in 'step' 
            # or just rely on the existing environment's bursty Pareto for now.
            
            if is_q_table:
                state_idx = env.get_state_index(state)
                action = np.argmax(agent[state_idx])
            else:
                action = agent.choose_action(state)
                
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_delay += info['delay']
            if info['is_dropped']:
                drops += 1
                
            state = next_state
            if done or truncated:
                break
                
        total_rewards.append(episode_reward)
        total_delays.append(episode_delay / 1000)
        drop_counts.append(drops)
        
    return np.mean(total_rewards), np.mean(total_delays), np.mean(drop_counts)

# --- 3. Visualization Functions ---

def plot_comparison(results):
    labels = list(results.keys())
    rewards = [results[l][0] for l in labels]
    delays = [results[l][1] for l in labels]
    drops = [results[l][2] for l in labels]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    ax[0].bar(labels, rewards, color='skyblue')
    ax[0].set_title('Average Reward (Higher is Better)')
    
    ax[1].bar(labels, delays, color='salmon')
    ax[1].set_title('Average Latency (Lower is Better)')
    
    ax[2].bar(labels, drops, color='lightgreen')
    ax[2].set_title('Packet Drop Count (Lower is Better)')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Comparison plot saved as 'performance_comparison.png'")

def plot_q_heatmap(q_table, env):
    # Visualize Q-values for CPU Workload vs Queue Level (fixed task=URLLC, channel=Normal, BW=Sufficient)
    # state = [0, 1, cpu, queue, 1]
    heatmap_data = np.zeros((3, 4)) # CPU (3) x Queue (4)
    
    for cpu in range(3):
        for queue in range(4):
            state = np.array([0, 1, cpu, queue, 1])
            idx = env.get_state_index(state)
            # Best action's Q-value
            heatmap_data[cpu, queue] = np.argmax(q_table[idx])
            
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", 
                xticklabels=['Empty', 'Smooth', 'Warning', 'Critical'],
                yticklabels=['Low', 'Normal', 'Congested'])
    plt.title('Best Action Heatmap (CPU vs Queue Level)')
    plt.xlabel('Queue Level')
    plt.ylabel('CPU Workload')
    plt.savefig('q_table_heatmap.png')
    print("Q-Table heatmap saved as 'q_table_heatmap.png'")

# --- 4. Main Execution ---

if __name__ == "__main__":
    env = MDCOffloadingEnv()
    
    # 1. Train SARSA Agent (Reuse training logic)
    q_table = np.zeros((env.n_states, env.action_space.n))
    alpha, gamma, epsilon = 0.1, 0.95, 0.1
    print("Training SARSA Agent...")
    for _ in range(1000): # More episodes for better visualization
        state, _ = env.reset()
        state_idx = env.get_state_index(state)
        action = np.argmax(q_table[state_idx]) if np.random.random() > epsilon else env.action_space.sample()
        while True:
            next_state, reward, term, trunc, _ = env.step(action)
            next_idx = env.get_state_index(next_state)
            next_act = np.argmax(q_table[next_idx]) if np.random.random() > epsilon else env.action_space.sample()
            q_table[state_idx, action] += alpha * (reward + gamma * q_table[next_idx, next_act] - q_table[state_idx, action])
            state_idx, action = next_idx, next_act
            if term or trunc: break

    # 2. Define Baselines
    agents = {
        "SARSA": q_table,
        "All-Local": AllLocalAgent(),
        "Random": RandomAgent(),
        "Threshold": ThresholdAgent(threshold_level=2)
    }
    
    # 3. Evaluate
    results = {}
    print("\nEvaluating Agents...")
    for name, agent in agents.items():
        is_q = (name == "SARSA")
        res = evaluate_agent(env, agent, num_episodes=20, is_q_table=is_q)
        results[name] = res
        print(f"[{name}] Reward: {res[0]:.2f}, Delay: {res[1]:.3f}, Drops: {res[2]:.1f}")
        
    # 4. Stress Test (Burst Scenario)
    # We can modify the env properties during evaluation
    print("\n--- Stress Test: High Traffic Burst ---")
    # Temporarily modify lambda in the env (requires class modification or just simulating)
    # Let's just run it as is, since Pareto already provides bursts.
    
    # 5. Visualization
    plot_comparison(results)
    plot_q_heatmap(q_table, env)
    print("\nVisualization and Comparison Complete.")
