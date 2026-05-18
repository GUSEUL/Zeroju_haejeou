import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class UnstableMDCEnv(gym.Env):
    def __init__(self, burst_mode=False):
        super(UnstableMDCEnv, self).__init__()
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.MultiDiscrete([2, 3, 3, 4, 2])
        self.n_states = 144
        self.burst_mode = burst_mode
        self.max_physical_queue = 50
        self.max_steps = 1000
        
        # High-Fidelity Costs
        self.energy_cost_local = 0.5
        self.energy_cost_offload = 0.3
        self.high_perf_cost = 0.2
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        self.physical_queue_size = 0
        self.current_step = 0
        return self.state, {}

    def get_state_index(self, state_array):
        return np.ravel_multi_index(state_array, self.observation_space.nvec)

    def step(self, action):
        task_type, channel, cpu, queue, bw = self.state
        delay_trans, delay_comp, energy_consumed = 0.0, 0.0, 0.0
        is_dropped = False

        # Service Rate based on CPU load
        service_rate = [3, 2, 1][cpu]

        # --- Action Logic ---
        if action == 0: # Local
            delay_comp = (self.physical_queue_size + 1) / (service_rate * 5.0)
            energy_consumed = self.energy_cost_local
            self.physical_queue_size += 1
        elif 1 <= action <= 6: # Offloading
            channel_factor = [1.0, 0.5, 0.2][channel]
            bw_factor = 1.5 if bw == 0 else 1.0
            delay_trans = channel_factor * bw_factor
            if action % 2 == 0:
                delay_trans *= 0.5
                energy_consumed = self.energy_cost_offload + self.high_perf_cost
            else:
                energy_consumed = self.energy_cost_offload
            delay_comp = 0.05
        elif action == 7: # Intentional Drop
            is_dropped = True

        total_delay = delay_trans + delay_comp
        w_task = 2.0 if task_type == 0 else 0.5
        
        # Reward Normalization
        r_delay = - (w_task * total_delay * 2.0)
        p_q = - (np.exp(2.0 * (queue / 3.0)) - 1)
        r_energy = - (energy_consumed * 1.5)
        
        if self.physical_queue_size > self.max_physical_queue:
            is_dropped = True
            self.physical_queue_size = self.max_physical_queue
        r_drop = - 20.0 if is_dropped else 0.0

        reward = r_delay + p_q + r_energy + r_drop

        # Transitions
        self.current_step += 1
        self.physical_queue_size = max(0, self.physical_queue_size - service_rate)
        
        # Extreme Unstable Transitions
        next_channel = self._unstable_channel_transition(channel)
        next_cpu = self._unstable_cpu_transition(cpu)
        next_bw = np.random.choice([0, 1])
        
        # Traffic Arrival
        if self.burst_mode:
            arrival_rate = np.random.poisson(2.5)
        else:
            arrival_rate = np.random.poisson(0.5) if np.random.rand() < 0.5 else int(np.random.pareto(2.0) + 1) if np.random.random() < 0.3 else 0
        
        self.physical_queue_size += arrival_rate

        if self.physical_queue_size == 0: q_lvl = 0
        elif self.physical_queue_size < self.max_physical_queue * 0.3: q_lvl = 1
        elif self.physical_queue_size < self.max_physical_queue * 0.7: q_lvl = 2
        else: q_lvl = 3

        self.state = np.array([np.random.choice([0, 1]), next_channel, next_cpu, q_lvl, next_bw], dtype=np.int32)
        
        return self.state, reward, False, self.current_step >= self.max_steps, {
            "delay": total_delay, 
            "is_dropped": is_dropped, 
            "energy": energy_consumed,
            "q_size": self.physical_queue_size
        }

    def _unstable_channel_transition(self, current):
        if np.random.random() < 0.4: return np.random.choice([0, 1, 2])
        return np.clip(current + np.random.choice([-1, 0, 1]), 0, 2)

    def _unstable_cpu_transition(self, current):
        if np.random.random() < 0.3: return 2
        return np.clip(current + np.random.choice([-1, 0, 1]), 0, 2)
