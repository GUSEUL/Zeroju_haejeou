# Active tracking test comment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math

class MDCMDPEnv(gym.Env):
    def __init__(self, arrival_lambda=None, reward_type="standard"):
        super(MDCMDPEnv, self).__init__()
        # 0:Local, 1:N1, 2:N2, 3:Pending
        self.action_space = spaces.Discrete(4)
        # Task(2), Comm_state(3), LocalQ(5), N1_Q(11), N2_Q(11)
        self.observation_space = spaces.MultiDiscrete([2, 3, 5, 11, 11])
        self.n_states = 2 * 3 * 5 * 11 * 11
        self.arrival_lambda = arrival_lambda if arrival_lambda is not None else 1.5
        self.max_steps = 1000
        self.reward_type = reward_type
        
        # Performance parameters
        self.service_rates = [2, 2, 2] # Local processing speed
        self.channel_factors = [1.5, 1.0, 0.5] # Transmission delay multiplier
        self.energy_costs = [0.8, 0.5, 0.3] # Energy per task
        
        # Reward Hyperparameters
        self.w = 0.6 # Weight for delay
        self.beta = 5.0 # Queue penalty scaling
        self.beta_neighbor = 5.0 # Neighbor queue penalty scaling
        self.gamma = 5.0 # Pending penalty (reduced to allow Pending to be optimal under high congestion)
        
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
        self.pending_buffer = 0 # Infinite pending buffer queue
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
        is_pending = False

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
            delay_comp = (self.neighbor_qs[idx] + 1) / 8.0 + 0.05
            self.neighbor_qs[idx] += 1
            if self.neighbor_qs[idx] >= 11:
                is_pending = True
                overflow_count = self.neighbor_qs[idx] - 10
                self.pending_buffer += overflow_count
                self.neighbor_qs[idx] = 10
        elif action == 3: # Pending
            is_pending = True
            self.pending_buffer += 1

        if self.local_q >= 5: # Overflow pending logic
            is_pending = True
            overflow_count = self.local_q - 4
            self.pending_buffer += overflow_count
            self.local_q = 4 # Clip to max index

        # --- Reward Calculation Logic ---
        total_delay = delay_trans + delay_comp
        w_task = 2.5 if task_type == 0 else 0.5
        
        # Fast min/max clip
        norm_delay = total_delay / self.max_delay
        if norm_delay < 0.0: norm_delay = 0.0
        elif norm_delay > 1.0: norm_delay = 1.0
        
        # Fast min/max clip
        norm_energy = energy_consumed / self.max_energy
        if norm_energy < 0.0: norm_energy = 0.0
        elif norm_energy > 1.0: norm_energy = 1.0
        
        cost_de = w_task * self.w * norm_delay + (1.0 - self.w) * norm_energy
        
        beta_task = 8.0 if task_type == 0 else 3.0
        beta_neighbor_task = 6.0 if task_type == 0 else 3.0
        
        if self.reward_type == "cliff":
            gamma_task = 1000.0
        else: # "standard"
            gamma_task = 30.0 if task_type == 0 else 10.0
            
        penalty_queue = beta_task * ((self.local_q / self.max_local_q) ** 2)
        penalty_pending = gamma_task if is_pending else 0.0
        penalty_neighbor = 0.0
        if action == 1 or action == 2:
            idx = action - 1
            penalty_neighbor = beta_neighbor_task * ((self.neighbor_qs[idx] / 10.0) ** 2)
            
        reward = -(cost_de + penalty_queue + penalty_pending + penalty_neighbor)
        # --------------------------

        # Transitions
        self.current_step += 1
        s_rate = self.service_rates[comm]
        self.local_q = max(0, self.local_q - s_rate)
        for i in range(2):
            served = 1 if random.random() < 0.2 else 0
            new_q = self.neighbor_qs[i] - served
            if new_q < 0:
                new_q = 0
            elif new_q > 10:
                new_q = 10
            self.neighbor_qs[i] = new_q

        next_task_type = 0 if random.random() < 0.5 else 1
        
        # True Poisson arrival sampling (uncapped, buffered)
        arrival = int(self.arrival_buffer[self.arrival_idx])
        self.arrival_idx = (self.arrival_idx + 1) % 100000
            
        # Background arrivals and overflow pending
        num_bg_pending = max(0, self.local_q + arrival - 4)
        bg_pending = num_bg_pending > 0
        self.local_q = min(4, self.local_q + arrival)
        self.pending_buffer += num_bg_pending
        
        # Apply pending penalty for background overflow
        if bg_pending:
            if self.reward_type == "cliff":
                reward -= 1000.0 * num_bg_pending
            else: # "standard"
                gamma_task = 30.0 if task_type == 0 else 10.0
                reward -= gamma_task * num_bg_pending
                
        # Replenish local queue from the infinite pending buffer if space becomes available
        if self.local_q < 4 and self.pending_buffer > 0:
            to_move = min(4 - self.local_q, self.pending_buffer)
            self.local_q += to_move
            self.pending_buffer -= to_move
                
        is_pending_overall = is_pending or bg_pending
        total_pending = (1 if is_pending else 0) + num_bg_pending

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
        
        return self.state, reward, False, self.current_step >= self.max_steps, {"delay": total_delay, "is_pending": is_pending_overall, "energy": energy_consumed, "pending_count": total_pending}
