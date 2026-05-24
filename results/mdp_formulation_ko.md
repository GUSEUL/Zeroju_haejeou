# MDP 태스크 오프로딩 수학적 수식 정의 (State, Action, Reward)

이 문서는 모바일 분산 컴퓨팅(Mobile Distributed Computing, MDC) 환경에서 태스크 오프로딩 최적화를 위해 설계된 MDP 모델의 상태 공간(State Space), 행동 공간(Action Space), 그리고 보상 함수(Reward Function)의 상세 수학적 정의를 기술합니다.

---

## 1. 상태 공간 (State Space, $S$)

시스템의 물리적 자원 상태와 채널 전송 품질 상태를 반영하며, 마르코프 성질(Markov Property)을 유지하도록 5개의 이산 변수의 결합으로 정의됩니다.

$$S = (Task\_Type, Comm\_State, Local\_Queue, Neighbor\_1\_Queue, Neighbor\_2\_Queue)$$

각 상태 구성 변수의 세부 이산 레벨은 다음과 같습니다.

1. **태스크 종류 ($Task\_Type \in \{0, 1\}$)**:
   - $Task = 0$: URLLC 태스크 (latency-sensitive, 지연 및 드롭에 매우 민감)
   - $Task = 1$: eMBB 태스크 (latency-tolerant, 대용량 전송 위주, 지연 허용)

2. **통신 및 채널 품질 ($Comm\_State \in \{0, 1, 2\}$)**:
   - $Comm = 0$: 불량 (Poor) - 낮은 전송 속도 및 로컬 연산 연계 지연 발생
   - $Comm = 1$: 보통 (Normal)
   - $Comm = 2$: 좋음 (Good) - 높은 전송 속도 보장

3. **로컬 대기열 크기 ($Local\_Queue \in \{0, 1, 2, 3, 4\}$)**:
   - 단말 장치의 물리 버퍼 용량은 $5$입니다. 큐 크기가 $5$ 이상 적재되면 강제로 드롭(Overflow Drop)이 발생하며, 인덱스로는 $0 \sim 4$로 표현됩니다.

4. **이웃 노드 1 대기열 크기 ($Neighbor\_1\_Queue \in \{0, 1, 2, \dots, 10\}$)**:
   - 무선 협력 통신 대상인 이웃 에지 노드 1의 물리 큐 크기 (버퍼 한도 10).

5. **이웃 노드 2 대기열 크기 ($Neighbor\_2\_Queue \in \{0, 1, 2, \dots, 10\}$)**:
   - 무선 협력 통신 대상인 이웃 에지 노드 2의 물리 큐 크기 (버퍼 한도 10).

$$\text{총 상태 공간 크기 } |S| = 2 \times 3 \times 5 \times 11 \times 11 = 3,630\text{개}$$

### 1.1 이웃 에지 노드 (Edge Device 1 & 2)의 특징 및 차이

본 MDP 모델링에서 무선 통신 파트너이자 오프로딩 대상인 **이웃 에지 노드 1 (Neighbor 1)**과 **이웃 에지 노드 2 (Neighbor 2)**는 단말기(Local Device)에 비해 다음과 같은 고유한 물리적 특징을 지니고 있습니다.

1. **상대적으로 높은 연산 능력 (High Computing Capability)**:
   - 단말기의 로컬 처리 속도 계수인 $ServiceRate \in \{1, 2, 3\}$와 달리, 이웃 에지 노드들은 각각 연산 능력 계수가 **$4.0$**으로 고정되어 있습니다. 
   - 따라서 이웃 노드 $i$로 태스크가 오프로딩되었을 때의 실제 계산 시간은 $Delay_{comp} = \frac{Q_{ni} + 1}{4.0} + 0.05$로 산출되어, 단말기 로컬 처리($\approx \frac{Q_{local}+1}{2\cdot ServiceRate}$)에 비해 큐당 대기 시간이 매우 짧습니다.

2. **여유로운 버퍼 용량 (Large Buffer Size)**:
   - 단말기의 로컬 버퍼 한도가 **$5$**로 매우 제한적인 데 반해, 이웃 노드들은 에지 서버급 자원을 보유하고 있어 버퍼 한도가 각각 **$10$**으로 2배 더 여유롭습니다.

3. **시간에 따른 동적 태스크 처리율 (Stochastic Service Rates)**:
   - 매 타임스텝마다 이웃 노드 1, 2의 대기열 크기($Q_{n1}, Q_{n2}$)는 에지 노드 자체의 태스크 처리로 인해 **$1$ 또는 $2$개만큼씩 무작위로 감소**합니다.
   - 이때 감소하는 처리율의 전이 확률은 다음과 같이 동일한 확률($50\%$)을 지닙니다.
     $$P(\text{처리율} = 1) = 0.5, \quad P(\text{처리율} = 2) = 0.5 \quad (\text{기대 처리율 } \mathbb{E}[\text{ServiceRate}_{neighbor}] = 1.5)$$

4. **대칭적 물리 특성과 부하 분산(Load Balancing) 필요성**:
   - 수학적 모델링 상 두 에지 노드는 동일한 연산 능력($4.0$)과 동일한 확률적 처리 감쇠($1.5$)를 가지며 대칭적입니다.
   - 하지만 에이전트 관점에서는 이웃 노드 1의 대기열 크기($Q_{n1}$)와 이웃 노드 2의 대기열 크기($Q_{n2}$)가 독립적인 차원으로 존재합니다. 따라서 에이전트는 무작위로 축적되는 두 큐의 현재 상태를 실시간으로 비교하여 더 한산한 노드로 태스크를 보내는 **동적 부하 분산(Load Balancing) 정책**을 스스로 학습하게 됩니다.

---

## 2. 행동 공간 (Action Space, $A$)

에이전트는 각 단계(Time-step)에서 유입된 태스크에 대해 다음 4가지 행동 중 하나를 결정합니다.

$$A = \{0, 1, 2, 3\}$$

* **$a = 0$ (Local Processing)**: 태스크를 로컬 CPU 대기열($Q_{local}$)에 삽입하여 직접 실행합니다.
* **$a = 1$ (Offload to Neighbor 1)**: 무선 링크를 통해 첫 번째 이웃 에지 노드 대기열($Q_{n1}$)로 오프로딩합니다.
* **$a = 2$ (Offload to Neighbor 2)**: 무선 링크를 통해 두 번째 이웃 에지 노드 대기열($Q_{n2}$)로 오프로딩합니다.
* **$a = 3$ (Intentional Drop)**: 버퍼 포화 상태를 막거나 지연 폭증을 줄이기 위해 태스크를 즉시 폐기(드롭)합니다.

---

## 3. 보상 함수 (Reward Function, $R$)

보상 함수는 시스템의 지연 시간 및 에너지 소모 비용($Cost_{delay\_energy}$)과 대기열 적재로 인한 혼잡 패널티($Penalty_{queue}$), 태스크 유실에 따르는 드롭 패널티($Penalty_{drop}$)의 총합에 음수부호($-$)를 붙여 정의됩니다. 에이전트는 누적 보상의 기댓값을 극대화(비용 최소화)하도록 유도됩니다.

$$R = -\left(Cost_{delay\_energy} + Penalty_{queue} + Penalty_{drop} + Penalty_{neighbor}\right)$$

### 3.1 지연 및 에너지 비용 ($Cost_{delay\_energy}$)

$$Cost_{delay\_energy} = w_{task} \cdot w_{delay} \cdot NormDelay + (1 - w_{delay}) \cdot NormEnergy$$

* **$w_{delay} = 0.6$**: 전체 비용 계산 중 지연 시간에 적용되는 가중치 비율
* **$NormDelay = \min\left(1.0, \frac{TotalDelay}{MaxDelay}\right)$**: 최대 5.0초를 기준으로 지연 시간을 $0 \sim 1$ 사이로 정규화 ($MaxDelay = 5.0$)
* **$NormEnergy = \min\left(1.0, \frac{EnergyConsumed}{MaxEnergy}\right)$**: 최대 1.0J을 기준으로 소모한 에너지를 $0 \sim 1$ 사이로 정규화 ($MaxEnergy = 1.0$)

**태스크별 긴급성 가중치 ($w_{task}$)**
* *Standard / Sparse / Cliff 설정*: URLLC ($Task=0$) 시 $2.0$, eMBB ($Task=1$) 시 $0.5$
* *Improved 설정 (개선 설계)*: URLLC ($Task=0$) 시 **$2.5$** (초저지연성 강화), eMBB ($Task=1$) 시 $0.5$

---

### 3.2 물리적 지연 및 에너지 소비 산출식

각 행동 $a$에 대해 소요되는 전송 지연($Delay_{trans}$), 컴퓨팅 연산 지연($Delay_{comp}$), 에너지($EnergyConsumed$)는 다음과 같습니다.

#### 1) 로컬 처리 ($a = 0$)
* 전송 지연: $Delay_{trans} = 0.0$
* 연산 지연: $Delay_{comp} = \frac{Q_{local} + 1}{2 \cdot ServiceRate(Comm)}$
  - $ServiceRate(Comm) \in \{1, 2, 3\}$ (채널 상태 $Comm \in \{0, 1, 2\}$에 비례하여 처리 속도 상승)
* 에너지 소모: $EnergyConsumed = EnergyCost(Comm) \in \{0.8, 0.5, 0.3\}$

#### 2) 이웃 노드 $i \in \{1, 2\}$ 오프로딩 ($a = 1, 2$)
* 전송 지연: $Delay_{trans} = ChannelFactor(Comm) \cdot 1.0$
  - $ChannelFactor(Comm) \in \{1.5, 1.0, 0.5\}$ (채널 품질 상태가 좋음(2)일 때 전송 지연이 가장 작음)
* 연산 지연: $Delay_{comp} = \frac{Q_{ni} + 1}{4.0} + 0.05$ (이웃 에지 노드는 속도 $4.0$을 가진 고속 프로세서 장착)
* 에너지 소모: $EnergyConsumed = EnergyCost(Comm) \cdot 0.6$ (오프로딩 시 무선 송출에 소모되는 전력은 로컬 연산 대비 $60\%$ 수준으로 절감)

#### 3) 의도적 드롭 ($a = 3$)
* 지연 시간 및 소모 에너지 모두 0으로 산출 ($Delay_{trans} = 0.0, Delay_{comp} = 0.0, EnergyConsumed = 0.0$)

---

### 3.3 대기열 혼잡 패널티 ($Penalty_{queue}$ & $Penalty_{neighbor}$)

버퍼가 포화 상태에 가까워질수록 패널티를 비선형적(제곱 형태)으로 부과하여 학습 불안정성을 예방하고 시스템 붕괴 임계점(Buffer Overflow)을 에이전트가 회피하도록 합니다.

#### 1) 로컬 대기열 패널티 ($Penalty_{queue}$)
$$Penalty_{queue} = \beta_{local} \cdot \left(\frac{Q_{local}}{MaxLocalQueue}\right)^2 \quad (\text{단, } MaxLocalQueue = 5.0)$$
* *Standard / Cliff 설정*: $\beta_{local} = 5.0$
* *Sparse 설정*: $\beta_{local} = 0.0$ (큐 패널티 미적용)
* *Improved 설정 (개선안)*: URLLC ($Task=0$) 시 $\beta_{local} = 8.0$ (대기 절대 불가), eMBB ($Task=1$) 시 $\beta_{local} = 3.0$ (버퍼 대기 수용)

#### 2) 이웃 대기열 패널티 ($Penalty_{neighbor}$)
에이전트가 오프로딩 $a \in \{1, 2\}$을 선택했을 때 적재 대상 이웃 노드 $i$의 버퍼 혼잡도를 반영합니다.
$$Penalty_{neighbor} = \beta_{neighbor} \cdot \left(\frac{Q_{ni}}{10.0}\right)^2$$
* *Standard / Cliff 설정*: $\beta_{neighbor} = 5.0$
* *Sparse 설정*: $\beta_{neighbor} = 0.0$
* *Improved 설정 (개선안)*: URLLC ($Task=0$) 시 $\beta_{neighbor} = 6.0$, eMBB ($Task=1$) 시 $\beta_{neighbor} = 3.0$

---

### 3.4 태스크 유실 패널티 ($Penalty_{drop}$)

에이전트가 의도적 드롭($a=3$)을 수행했거나, 타임스텝 연산 이후 백그라운드 유입 Poisson 도착(Poisson Arrival)량으로 인해 로컬 큐 용량을 초과해 자동 드롭($num\_bg\_drops$)된 경우에 가해집니다.

* **의도적 드롭 ($a=3$) 패널티**: $\gamma$
* **백그라운드 버퍼 오버플로우 패널티**: $\gamma \cdot num\_bg\_drops$

**보상 설정 방식별 패널티 상수 ($\gamma$)**
1. **Standard 설정**: $\gamma = 5.0$ (낮은 패널티로 인하여 중부하 조건에서 쉽게 태스크를 버림)
2. **Sparse 설정**: $\gamma = 100.0$ (비용을 배제하고 드롭 여부에만 집중)
3. **Cliff 설정**: $\gamma = 1000.0$ (드롭을 매우 치명적인 에러로 간주)
4. **Improved 설정 (개선안)**: **태스크 종류별 차등 가점**
   - URLLC ($Task=0$) 시: $\gamma_{task} = 30.0$ (지연 민감 태스크의 전송 유실 원천 방지)
   - eMBB ($Task=1$) 시: $\gamma_{task} = 10.0$ (혼잡 상황 시 효율적인 드롭 타협 가능)

---

## 4. 학습 알고리즘 개선 사항 (Pessimistic Q-Initialization & Exploration Tuning)

음수 보상 구조를 가진 본 오프로딩 MDP에서 미방문 상태 및 초기 탐험 시의 오작동(비어 있는 대기열에서의 무작위 드롭 등)을 해결하기 위해 알고리즘 파라미터를 다음과 같이 수정하였습니다.

### 4.1 비관적 Q-테이블 초기화 (Pessimistic Q-table Initialization)
* **기존**: $Q(s, a) = 0.0$으로 전체 초기화
  - 모든 정상 보상이 음수($< 0$)이므로, 한 번도 경험하지 않은 나쁜 행동(예: 드롭 $a=3$)의 Q-값이 경험하여 패널티를 받은 안전한 행동(로컬 처리 $a=0$)의 Q-값($\approx -3.5$)보다 크게 평가되어 최적화 도중 오동작을 일으킴.
* **수정**: $Q(s, a) = -150.0$으로 초기치 설정 (비관적 초기화)
  - 학습되지 않은 미방문 상태-행동 쌍의 가치를 매우 낮게 평가함으로써, 검증되지 않은 돌발 행동을 취하는 문제를 원천적으로 방지하고 충분한 탐험 이후 검증된 최적의 경로를 우선적으로 선택하도록 유도함.

### 4.2 학습 시간 연장 및 탐험 감쇠율 완화 (Exploration Scheduling Tuning)
* **기존**: 5,000 에피소드 학습 및 총 에피소드의 $40\%$ 시점까지 $\epsilon$-greedy 탐험율 감쇠 진행
* **수정**: **20,000 에피소드** 학습으로 연장 및 감쇠 스케줄러 완화 (총 에피소드의 **$60\%$** 시점까지 서서히 감쇠)
  - 학습 수렴 횟수를 4배 늘려 희소 상태 공간(State Sparsity)의 방문 빈도를 확보함.
  - 탐험 확률 $\epsilon$이 더 오랜 기간 동안 큰 값을 유지하도록 스케줄링을 완화하여, 에이전트가 벼랑 끝 상태나 드롭 상태의 정확한 패널티 기대치를 충분히 업데이트할 시간을 확보함.

이 수정을 통해 Q-Learning과 Expected SARSA 에이전트는 미방문 상태에서의 오작동(비어 있는 대기열에서의 태스크 드롭)을 근절하고, 수학적 상한선인 DP(동적 계획법) 정책의 최적 보상 수준과 드롭 성능에 완벽하게 수렴하였습니다.

