import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MDCMDPEnv(gym.Env):
    """MDC task offloading MDP.

    state[1] ("Sys_State") is a joint index that captures both channel
    quality and local CPU contention; the same index drives transmission
    factors and local service rate. See MDP_DESIGN_REVISIONS.md for the
    justification of this design.
    """
    def __init__(self, arrival_lambda=None, reward_type="standard"):
        super(MDCMDPEnv, self).__init__()
        # 0:Local, 1:N1, 2:N2, 3:Drop
        self.action_space = spaces.Discrete(4)
        # Task(2), Sys_State(3), LocalQ(5), N1_Q(11), N2_Q(11)
        self.observation_space = spaces.MultiDiscrete([2, 3, 5, 11, 11])
        self.n_states = 2 * 3 * 5 * 11 * 11
        self.arrival_lambda = arrival_lambda if arrival_lambda is not None else 1.5
        self.max_steps = 1000
        self.reward_type = reward_type

        # Performance parameters (indexed by sys_state: 0:Poor, 1:Normal, 2:Good)
        self.service_rates = [1, 2, 3]       # Local CPU service rate
        self.channel_factors = [1.5, 1.0, 0.5]  # Transmission delay multiplier
        self.energy_costs = [0.8, 0.5, 0.3]  # Energy per task (local compute)

        # Reward Hyperparameters
        self.w = 0.6              # delay/energy trade-off
        self.beta = 5.0           # local queue penalty scale
        self.beta_n = 3.0         # per-neighbor queue penalty scale
        self.gamma = 20.0         # drop penalty
        self.success_bonus = 1.0  # base bonus for successful processing (standard only)

        # Normalization factors
        self.max_delay = 5.0
        self.max_energy = 1.0
        self.max_local_q = 5.0
        self.max_neighbor_q = 10.0

        # Background load at neighbor nodes (other users / cells)
        self.neighbor_bg_lambda = 0.7

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.local_q = 0
        self.neighbor_qs = np.zeros(2, dtype=np.int32)
        self.sys_state = 1
        self.current_step = 0
        self.state = np.array([0, 1, 0, 0, 0], dtype=np.int32)
        return self.state, {}

    def get_state_index(self, state_array):
        return np.ravel_multi_index(state_array, self.observation_space.nvec)

    def step(self, action):
        task_type, sys_state, q_local, q_n1, q_n2 = self.state

        delay_trans, delay_comp = 0.0, 0.0
        energy_consumed = 0.0
        is_dropped = False
        is_success = False

        if action == 0:  # Local
            if self.local_q >= 4:
                # Local queue full: reject without charging compute cost
                is_dropped = True
            else:
                s_rate = self.service_rates[sys_state]
                delay_comp = (self.local_q + 1) / (s_rate * 2.0)
                energy_consumed = self.energy_costs[sys_state]
                self.local_q += 1
                is_success = True
        elif action == 1 or action == 2:  # Offload to neighbor
            idx = action - 1
            if self.neighbor_qs[idx] >= 10:
                # Target neighbor full: reject without charging tx cost
                is_dropped = True
            else:
                chan_factor = self.channel_factors[sys_state]
                delay_trans = chan_factor * 1.0
                energy_consumed = self.energy_costs[sys_state] * 0.6
                delay_comp = (self.neighbor_qs[idx] + 1) / 4.0 + 0.05
                self.neighbor_qs[idx] += 1
                is_success = True
        elif action == 3:  # Intentional drop
            is_dropped = True

        # --- Reward Calculation Logic ---
        total_delay = delay_trans + delay_comp
        w_task = 2.0 if task_type == 0 else 0.5

        if self.reward_type == "sparse":
            # Sparse: only care about drops
            reward = -100.0 if is_dropped else -0.1

        elif self.reward_type == "cliff":
            # Cliff: huge drop penalty + edge noise (kept intentionally
            # minimalist to expose SARSA's risk aversion)
            norm_delay = np.clip(total_delay / self.max_delay, 0.0, 1.0)
            norm_energy = np.clip(energy_consumed / self.max_energy, 0.0, 1.0)
            cost_de = w_task * self.w * norm_delay + (1.0 - self.w) * norm_energy

            penalty_drop = 1000.0 if is_dropped else 0.0
            reward = -(cost_de + penalty_drop)

            if self.local_q >= 4 and not is_dropped:
                reward += np.random.normal(0, 5.0)

        else:  # "standard"
            norm_delay = np.clip(total_delay / self.max_delay, 0.0, 1.0)
            norm_energy = np.clip(energy_consumed / self.max_energy, 0.0, 1.0)
            cost_delay_energy = w_task * self.w * norm_delay + (1.0 - self.w) * norm_energy

            # System-wide queue congestion: local + both neighbors
            penalty_queue = (
                self.beta * (self.local_q / self.max_local_q) ** 2
                + self.beta_n * (self.neighbor_qs[0] / self.max_neighbor_q) ** 2
                + self.beta_n * (self.neighbor_qs[1] / self.max_neighbor_q) ** 2
            )

            penalty_drop = self.gamma if is_dropped else 0.0
            success_term = (w_task * self.success_bonus) if is_success else 0.0

            reward = -(cost_delay_energy + penalty_queue + penalty_drop) + success_term
        # --------------------------

        # Transitions
        self.current_step += 1
        s_rate = self.service_rates[sys_state]
        self.local_q = max(0, self.local_q - s_rate)
        for i in range(2):
            # Neighbor service
            self.neighbor_qs[i] = max(0, self.neighbor_qs[i] - np.random.randint(1, 3))
            # Background arrivals from other users
            bg_arrival = np.random.poisson(self.neighbor_bg_lambda)
            self.neighbor_qs[i] = np.clip(self.neighbor_qs[i] + bg_arrival, 0, 10)

        next_task_type = np.random.choice([0, 1])
        arrival = np.random.poisson(self.arrival_lambda)
        self.local_q = np.clip(self.local_q + arrival, 0, 4)

        next_sys_state = np.clip(sys_state + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]), 0, 2)
        self.state = np.array([next_task_type, next_sys_state, self.local_q, self.neighbor_qs[0], self.neighbor_qs[1]], dtype=np.int32)

        return self.state, reward, False, self.current_step >= self.max_steps, {
            "delay": total_delay,
            "is_dropped": is_dropped,
            "energy": energy_consumed,
            "is_success": is_success,
        }
