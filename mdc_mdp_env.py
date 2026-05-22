import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math

class MDCMDPEnv(gym.Env):
    def __init__(self, arrival_lambda=None, reward_type="standard"):
        super(MDCMDPEnv, self).__init__()
        # 0:Local, 1:N1, 2:N2, 3:Drop
        self.action_space = spaces.Discrete(4)
        # Task(2), Comm_state(3), LocalQ(5), N1_Q(11), N2_Q(11)
        self.observation_space = spaces.MultiDiscrete([2, 3, 5, 11, 11])
        self.n_states = 2 * 3 * 5 * 11 * 11
        self.arrival_lambda = arrival_lambda if arrival_lambda is not None else 1.5
        self.max_steps = 1000
        self.reward_type = reward_type
        
        # Performance parameters
        self.service_rates = [1, 2, 3] # Local processing speed
        self.channel_factors = [1.5, 1.0, 0.5] # Transmission delay multiplier
        self.energy_costs = [0.8, 0.5, 0.3] # Energy per task
        
        # Reward Hyperparameters
        self.w = 0.6 # Weight for delay
        self.beta = 5.0 # Queue penalty scaling
        self.beta_neighbor = 5.0 # Neighbor queue penalty scaling
        self.gamma = 5.0 # Drop penalty (reduced to allow Drop to be optimal under high congestion)
        
        # Normalization factors
        self.max_delay = 5.0 
        self.max_energy = 1.0
        self.max_local_q = 5.0

        # Pre-generated Poisson arrivals for speed
        self.arrival_buffer = np.random.poisson(self.arrival_lambda, size=100000)
        self.arrival_idx = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.arrival_idx = 0
        if options and options.get("random_start", False):
            self.local_q = random.randint(0, 4)
            self.neighbor_qs = [random.randint(0, 10), random.randint(0, 10)]
            self.comm = random.randint(0, 2)
            task_type = random.randint(0, 1)
            self.state = np.array([task_type, self.comm, self.local_q, self.neighbor_qs[0], self.neighbor_qs[1]], dtype=np.int32)
        else:
            self.local_q = 0
            self.neighbor_qs = [0, 0]
            self.comm = 1
            self.state = np.array([0, 1, 0, 0, 0], dtype=np.int32)
        self.current_step = 0
        return self.state, {}

    def get_state_index(self, state_array):
        return int(state_array[0] * 1815 + state_array[1] * 605 + state_array[2] * 121 + state_array[3] * 11 + state_array[4])

    def step(self, action):
        task_type, comm, q_local, q_n1, q_n2 = self.state
        
        delay_trans, delay_comp = 0.0, 0.0
        energy_consumed = 0.0
        is_dropped = False

        if action == 0: # Local
            s_rate = self.service_rates[comm]
            delay_comp = (self.local_q + 1) / (s_rate * 2.0)
            energy_consumed = self.energy_costs[comm]
            self.local_q += 1
        elif action == 1 or action == 2: # Offload
            idx = action - 1
            chan_factor = self.channel_factors[comm]
            delay_trans = chan_factor * 1.0
            energy_consumed = self.energy_costs[comm] * 0.6
            delay_comp = (self.neighbor_qs[idx] + 1) / 4.0 + 0.05
            self.neighbor_qs[idx] += 1
        elif action == 3: # Drop
            is_dropped = True

        if self.local_q >= 5: # Overflow drop logic
            is_dropped = True
            self.local_q = 4 # Clip to max index

        # --- Reward Calculation Logic ---
        total_delay = delay_trans + delay_comp
        w_task = 2.0 if task_type == 0 else 0.5
        
        # Fast min/max clip
        norm_delay = total_delay / self.max_delay
        if norm_delay < 0.0: norm_delay = 0.0
        elif norm_delay > 1.0: norm_delay = 1.0
        
        norm_energy = energy_consumed / self.max_energy
        if norm_energy < 0.0: norm_energy = 0.0
        elif norm_energy > 1.0: norm_energy = 1.0
        
        cost_de = w_task * self.w * norm_delay + (1.0 - self.w) * norm_energy
        
        if self.reward_type == "sparse":
            # Sparse: Only care about drops, very small step penalty, plus mild regularization
            reward = -100.0 if is_dropped else (-0.1 - 0.01 * cost_de)
            
        elif self.reward_type == "cliff":
            # Cliff: Huge penalty for drop, and noise near the cliff (local_q=4)
            penalty_drop = 1000.0 if is_dropped else 0.0
            penalty_neighbor = 0.0
            if action == 1 or action == 2:
                idx = action - 1
                penalty_neighbor = self.beta_neighbor * ((self.neighbor_qs[idx] / 10.0) ** 2)
            reward = -(cost_de + penalty_drop + penalty_neighbor)
            
            if self.local_q >= 4 and not is_dropped:
                # Add high variance noise near the edge to scare SARSA
                reward += random.normalvariate(0, 5.0)
                
        else: # "standard"
            penalty_queue = self.beta * ((self.local_q / self.max_local_q) ** 2)
            penalty_drop = self.gamma if is_dropped else 0.0
            penalty_neighbor = 0.0
            if action == 1 or action == 2:
                idx = action - 1
                penalty_neighbor = self.beta_neighbor * ((self.neighbor_qs[idx] / 10.0) ** 2)
            reward = - (cost_de + penalty_queue + penalty_drop + penalty_neighbor)
        # --------------------------

        # Transitions
        self.current_step += 1
        s_rate = self.service_rates[comm]
        self.local_q = max(0, self.local_q - s_rate)
        for i in range(2):
            served = 1 if random.random() < 0.5 else 2
            new_q = self.neighbor_qs[i] - served
            if new_q < 0:
                new_q = 0
            self.neighbor_qs[i] = new_q

        next_task_type = 0 if random.random() < 0.5 else 1
        
        # True Poisson arrival sampling (uncapped, buffered)
        arrival = int(self.arrival_buffer[self.arrival_idx])
        self.arrival_idx = (self.arrival_idx + 1) % 100000
            
        # Background arrivals and overflow drops
        num_bg_drops = max(0, self.local_q + arrival - 4)
        bg_dropped = num_bg_drops > 0
        self.local_q = min(4, self.local_q + arrival)
        
        # Apply drop penalty for background overflow
        if bg_dropped:
            if self.reward_type == "sparse":
                reward -= 100.0 * num_bg_drops
            elif self.reward_type == "cliff":
                reward -= 1000.0 * num_bg_drops
            else: # standard
                reward -= self.gamma * num_bg_drops
                
        is_dropped_overall = is_dropped or bg_dropped
        total_drops = (1 if is_dropped else 0) + num_bg_drops

        r_comm = random.random()
        if r_comm < 0.1:
            change = -1
        elif r_comm > 0.9:
            change = 1
        else:
            change = 0
        next_comm = comm + change
        if next_comm < 0:
            next_comm = 0
        elif next_comm > 2:
            next_comm = 2
            
        self.state = np.array([next_task_type, next_comm, self.local_q, self.neighbor_qs[0], self.neighbor_qs[1]], dtype=np.int32)
        
        return self.state, reward, False, self.current_step >= self.max_steps, {"delay": total_delay, "is_dropped": is_dropped_overall, "energy": energy_consumed, "dropped_count": total_drops}
