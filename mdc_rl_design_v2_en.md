# Design of Reinforcement Learning Model for Communication Network Offloading based on Micro Data Center (MDC) (Revised Edition)

## 1. Project Overview and Environment Setup
* **Optimization Goal:** To learn ultra-lightweight Task Offloading and resource allocation policies that utilize limited resources of 5G/6G base station Micro Data Centers (MDC), satisfy requirements per traffic characteristic, and preemptively prevent packet drops.
* **Learning Optimization Strategy:** Considering the physical computation and memory constraints of Edge equipment, **Deep Reinforcement Learning (Deep RL) is excluded**. Instead, a **Tabular Model-free RL** approach is maintained, enabling ultra-fast lookups in microseconds ($\mu s$).
* **Temporal Characteristics:** Defined as a **Continuing Task** that operates 24/7 without interruption. A discount factor ($\gamma = 0.95 \sim 0.99$) is applied to prevent infinite divergence of rewards and to reflect future ripple effects.

---

## 2. Revised Markov Decision Process (MDP) Design

### 2.1 $\mathcal{S}$: State Space
Moving away from excessive simplification, traffic heterogeneity and queue risk levels have been refined. By integrating redundant or low-impact variables, the total number of states is capped at **144** to defend against the curse of dimensionality.

| Category | State Variable | Discretization Levels | Number of Cases | Integration Intent |
| :--- | :--- | :--- | :---: | :--- |
| **Traffic** | Task Type | URLLC (Ultra-Reliable Low-Latency), eMBB (Enhanced Mobile Broadband) | 2 | Providing customized routing indicators per service nature |
| **Network** | Integrated Channel Status | Poor (0), Normal (1), Good (2) | 3 | Merging existing CQI and interference levels into a single effective transmission rate |
| **Server (MDC)** | Server CPU Workload | Low (0), Normal (1), Congested (2) | 3 | For predicting computational latency of local and neighboring MDCs |
| **Server (MDC)** | **Queue Level** | **Empty (0), Smooth (1), Warning (2), Critical (3)** | 4 | 4-stage division for preemptive load balancing before the queue overflows |
| **Network** | Local Available Bandwidth | Insufficient (0), Sufficient (1) | 2 | Checking physical resource constraints of the transmission link |

* **Total States:** $2 \times 3 \times 3 \times 4 \times 2 = 144$

---

### 2.2 $\mathcal{A}$: Action Space
Beyond simple relaying, **'Local Execution'** within the MDC and **'Intentional Drop (Admission Control)'** to prevent network collapse have been added for increased realism. (Total of 8 Actions)

1. **Local Execution:**
   * `Action 0`: Immediate computational processing at the receiving MDC without transmitting to other base stations (Transmission delay $0$, only computational delay occurs).
2. **Neighboring Edge Offloading & Joint Bandwidth Allocation:**
   * `Action 1`: Offload to Edge 1 + Normal Bandwidth Allocation
   * `Action 2`: Offload to Edge 1 + Concentrated Bandwidth Allocation
   * `Action 3`: Offload to Edge 2 + Normal Bandwidth Allocation
   * `Action 4`: Offload to Edge 2 + Concentrated Bandwidth Allocation
   * `Action 5`: Offload to Edge 3 + Normal Bandwidth Allocation
   * `Action 6`: Offload to Edge 3 + Concentrated Bandwidth Allocation
3. **Flow Control:**
   * `Action 7`: Intentional Drop (Preemptive entry control to prevent paralysis of the entire network and queue).

---

### 2.3 $\mathcal{R}$: Reward Function
The issues of ambiguity in weight tuning and learning paralysis due to Sparse Penalties have been completely revamped using **Reward Shaping** techniques.

$$r_t = - \left( w_{task} \cdot D_{total} \right) - P(q_t) - C_{drop}$$

* **$D_{total}$ (Total Delay Penalty):** Physical sum of transmission delay ($D_{trans}$) and computational delay ($D_{comp}$).
* **$w_{task}$ (Traffic Weight):** Corresponds to the 'Task Type' of the current state. Applied differentially, e.g., $w_{task} = 2.0$ for URLLC where ultra-low latency is critical, and $w_{task} = 0.5$ for eMBB which is relatively tolerant to latency, encouraging the agent to understand service characteristics.
* **$P(q_t)$ (Early Warning Queue Penalty):** A penalty that increases exponentially proportional to the queue length even before it overflows completely.
  $$P(q_t) = e^{k \cdot \left( \frac{\text{Current Queue Size}}{\text{Max Queue Capacity}} \right)} - 1$$
  Through this, when the queue level reaches 'Warning (2)' or 'Critical (3)', the agent experiences extreme negative rewards and preemptively allocates more resources or bypasses to neighboring MDCs.
* **$C_{drop}$ (Fatal Drop Penalty):** A strong fixed penalty imposed only when packets are forcibly lost due to capacity overflow or when an intentional drop (`Action 7`) is chosen.

---

## 3. Algorithm Application and Engineering Review

* **Q-Table Structure:** Maintains only a 2D matrix of $144 \times 8 = 1,152$ entries. Memory requirement is less than a few KB, allowing it to be hard-coded into edge communication equipment chipsets.
* **Critical Adoption and Tuning Suggestions for SARSA Technique:**
  * **Conservatism Control:** Due to the On-policy nature of SARSA, there is a risk that the policy becomes overly conservative by the amount of exploration ($\epsilon$), leading to underutilization of resources and frequent premature drops.
  * **Solution:** Set exploration rate $\epsilon$ relatively high at the beginning of learning, but apply a steep $\epsilon\text{-decay}$ schedule as episodes (steps) progress to settle into the optimal policy. When deploying to actual physical equipment after stabilization in a simulator environment, $\epsilon$ should be fixed to 0 or a minimal value ($0.01$) to push resource utilization to its limit.
