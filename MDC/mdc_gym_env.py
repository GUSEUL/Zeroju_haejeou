import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class MDCOffloadingEnv(gym.Env):
    def __init__(self, burst_mode=False, arrival_lambda=None):
        super(MDCOffloadingEnv, self).__init__()
        # State: Task(2), Channel(3), CPU(3), Queue(4), BW(2), Best_N_ID(6), Best_N_Q(4)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.MultiDiscrete([2, 3, 3, 4, 2, 6, 4])
        self.n_states = 3456 # 144 * 6 * 4
        self.burst_mode = burst_mode
        self.arrival_lambda = arrival_lambda # Custom traffic load
        
        # Physical Constants
        self.max_physical_queue = 50
        self.max_steps = 1000
        
        # Resource Costs (Energy & Performance)
        self.energy_cost_local = 0.5    # J per task
        self.energy_cost_offload = 0.3  # J per task (Transmission energy)
        self.high_perf_cost = 0.2       # Additional cost for concentrated bandwidth
        
        self.reset()

    def _get_neighbor_info(self):
        """Helper to find the best neighbor (min queue)."""
        # Convert physical queues to levels 0-3
        levels = []
        for q in self.neighbor_queues:
            if q == 0: levels.append(0)
            elif q < self.max_physical_queue * 0.3: levels.append(1)
            elif q < self.max_physical_queue * 0.7: levels.append(2)
            else: levels.append(3)
        
        best_id = np.argmin(self.neighbor_queues)
        best_q = levels[best_id]
        return best_id, best_q

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.physical_queue_size = 0
        self.neighbor_queues = np.zeros(6, dtype=np.int32)
        self.current_step = 0
        
        best_id, best_q = self._get_neighbor_info()
        # Initial State with neighbor info
        self.state = np.array([0, 1, 1, 0, 1, best_id, best_q], dtype=np.int32)
        return self.state, {}

    def get_state_index(self, state_array):
        return np.ravel_multi_index(state_array, self.observation_space.nvec)

    def step(self, action):
        task_type, channel, cpu, queue, bw, _, _ = self.state
        delay_trans, delay_comp = 0.0, 0.0
        energy_consumed = 0.0
        is_dropped = False

        # Update Neighbor Queues (Processing at neighbors)
        for i in range(6):
            self.neighbor_queues[i] = max(0, self.neighbor_queues[i] - np.random.randint(1, 4))

        # --- 1. Realistic Dynamics: Service Rate ---
        service_rate = [3, 2, 1][cpu]

        # --- 2. Action Execution ---
        if action == 0: # Local Processing
            delay_comp = (self.physical_queue_size + 1) / (service_rate * 5.0) 
            energy_consumed = self.energy_cost_local
            self.physical_queue_size += 1
            
        elif 1 <= action <= 6: # Offloading
            idx = action - 1
            channel_factor = [1.0, 0.5, 0.2][channel]
            bw_factor = 1.5 if bw == 0 else 1.0
            delay_trans = channel_factor * bw_factor
            
            if action % 2 == 0:
                delay_trans *= 0.5
                energy_consumed = self.energy_cost_offload + self.high_perf_cost
            else:
                energy_consumed = self.energy_cost_offload
            
            # Dynamic Remote Delay based on the chosen neighbor's queue
            self.neighbor_queues[idx] += 1
            delay_comp = (self.neighbor_queues[idx]) / 10.0 + 0.05 

        elif action == 7: # Intentional Drop
            is_dropped = True

        total_delay = delay_trans + delay_comp

        # --- 3. Reward Calculation ---
        w_task = 2.0 if task_type == 0 else 0.5
        r_delay = - (w_task * total_delay * 2.0)
        p_q = - (np.exp(2.0 * (queue / 3.0)) - 1)
        r_energy = - (energy_consumed * 1.5)
        
        if self.physical_queue_size > self.max_physical_queue:
            is_dropped = True
            self.physical_queue_size = self.max_physical_queue
            
        r_drop = - 20.0 if is_dropped else 0.0
        reward = r_delay + p_q + r_energy + r_drop

        # --- 4. State Transitions ---
        self.current_step += 1
        self.physical_queue_size = max(0, self.physical_queue_size - service_rate)
        
        # Traffic Arrival
        next_task_type = np.random.choice([0, 1])
        if self.arrival_lambda is not None:
            arrival_rate = np.random.poisson(self.arrival_lambda)
        elif self.burst_mode:
            arrival_rate = np.random.poisson(2.5)
        else:
            arrival_rate = np.random.poisson(0.5) if next_task_type == 0 else (int(np.random.pareto(2.0) + 1) if np.random.random() < 0.3 else 0)
        
        self.physical_queue_size += arrival_rate

        # Markov Transitions
        next_channel = self._transition_state(channel, 3)
        next_cpu = self._transition_state(cpu, 3)
        next_bw = self._transition_state(bw, 2)
        
        if self.physical_queue_size == 0: q_lvl = 0
        elif self.physical_queue_size < self.max_physical_queue * 0.3: q_lvl = 1
        elif self.physical_queue_size < self.max_physical_queue * 0.7: q_lvl = 2
        else: q_lvl = 3

        # Update Best Neighbor Info for the next state
        best_id, best_q = self._get_neighbor_info()

        self.state = np.array([next_task_type, next_channel, next_cpu, q_lvl, next_bw, best_id, best_q], dtype=np.int32)
        
        return self.state, reward, False, self.current_step >= self.max_steps, {
            "delay": total_delay, 
            "is_dropped": is_dropped, 
            "q_size": self.physical_queue_size,
            "energy": energy_consumed
        }

    def _transition_state(self, current, n_levels):
        change = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
        return np.clip(current + change, 0, n_levels - 1)
