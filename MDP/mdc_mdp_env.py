import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MDCMDPEnv(gym.Env):
    def __init__(self, arrival_lambda=None):
        super(MDCMDPEnv, self).__init__()
        # 0:Local, 1:N1, 2:N2, 3:Drop
        self.action_space = spaces.Discrete(4)
        # Task(2), Comm_state(3), LocalQ(5), N1_Q(11), N2_Q(11)
        # LocalQ capacity: 5 (0-4)
        # NeighborQ capacity: 11 (0-10)
        self.observation_space = spaces.MultiDiscrete([2, 3, 5, 11, 11])
        self.n_states = 2 * 3 * 5 * 11 * 11 # 3630 states
        self.arrival_lambda = arrival_lambda if arrival_lambda is not None else 1.5
        self.max_steps = 1000
        
        # Performance parameters based on Comm_state
        self.service_rates = [1, 2, 3] # Local processing speed
        self.channel_factors = [1.5, 1.0, 0.5] # Transmission delay multiplier
        self.energy_costs = [0.8, 0.5, 0.3] # Energy per task
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.local_q = 0
        self.neighbor_qs = np.zeros(2, dtype=np.int32)
        self.comm_state = 1
        self.current_step = 0
        self.state = np.array([0, 1, 0, 0, 0], dtype=np.int32) 
        return self.state, {}

    def get_state_index(self, state_array):
        return np.ravel_multi_index(state_array, self.observation_space.nvec)

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

        total_delay = delay_trans + delay_comp
        w_task = 2.0 if task_type == 0 else 0.5
        
        r_delay = - (w_task * total_delay * 2.0)
        # Queue Penalty normalized by capacity (5)
        p_q = - (np.exp(2.0 * (self.local_q / 5.0)) - 1)
        r_energy = - (energy_consumed * 1.5)
        
        if self.local_q >= 5: # Overflow drop
            is_dropped = True
            self.local_q = 4 # Clip to max index
        r_drop = - 20.0 if is_dropped else 0.0
        
        reward = r_delay + p_q + r_energy + r_drop

        # Transitions
        self.current_step += 1
        s_rate = self.service_rates[comm]
        self.local_q = max(0, self.local_q - s_rate)
        for i in range(2):
            self.neighbor_qs[i] = max(0, self.neighbor_qs[i] - np.random.randint(1, 3))
            self.neighbor_qs[i] = np.clip(self.neighbor_qs[i], 0, 10)

        next_task_type = np.random.choice([0, 1])
        arrival = np.random.poisson(self.arrival_lambda)
        self.local_q = np.clip(self.local_q + arrival, 0, 4)

        next_comm = np.clip(comm + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]), 0, 2)
        self.state = np.array([next_task_type, next_comm, self.local_q, self.neighbor_qs[0], self.neighbor_qs[1]], dtype=np.int32)
        
        return self.state, reward, False, self.current_step >= self.max_steps, {"delay": total_delay, "is_dropped": is_dropped, "energy": energy_consumed}
