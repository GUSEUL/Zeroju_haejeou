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
   - $Comm = 0$: 불량 (Poor) - 낮은 전송 속도로 인한 전송 지연 발생 (로컬 연산 속도는 독립적으로 고정됨)
   - $Comm = 1$: 보통 (Normal)
   - $Comm = 2$: 좋음 (Good) - 높은 전송 속도로 인한 빠른 전송 보장

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
   - 단말기의 로컬 처리 속도 계수인 $ServiceRate = 2$와 달리, 이웃 에지 노드들은 각각 연산 능력 계수(분모)가 **$8.0$**으로 설정되어 있습니다. 
   - 따라서 이웃 노드 $i$로 태스크가 오프로딩되었을 때의 실제 계산 시간은 $Delay_{comp} = \frac{Q_{ni} + 1}{8.0} + 0.05$로 산출되어, 단말기 로컬 처리($Delay_{comp} = \frac{Q_{local} + 1}{2 \cdot ServiceRate} = \frac{Q_{local} + 1}{4.0}$)에 비해 큐당 대기 시간이 매우 짧습니다.

2. **여유로운 버퍼 용량 (Large Buffer Size)**:
   - 단말기의 로컬 버퍼 한도가 **$5$**로 매우 제한적인 데 반해, 이웃 노드들은 에지 서버급 자원을 보유하고 있어 버퍼 한도가 각각 **$10$**으로 2배 더 여유롭습니다.

3. **시간에 따른 동적 태스크 처리율 (Stochastic Service Rates)**:
   - 매 타임스텝마다 이웃 노드 1, 2의 대기열 크기($Q_{n1}, Q_{n2}$)는 에지 노드 자체의 태스크 처리로 인해 **$1$ 또는 $2$개만큼씩 무작위로 감소**합니다.
   - 이때 감소하는 처리율의 전이 확률은 다음과 같이 동일한 확률($50\%$)을 지닙니다.
     $$P(\text{처리율} = 1) = 0.5, \quad P(\text{처리율} = 2) = 0.5 \quad (\text{기대 처리율 } \mathbb{E}[\text{ServiceRate}_{neighbor}] = 1.5)$$

4. **대칭적 물리 특성과 부하 분산(Load Balancing) 필요성**:
   - 수학적 모델링 상 두 에지 노드는 동일한 연산 능력 계수($8.0$)와 동일한 확률적 처리 감쇠($1.5$)를 가지며 대칭적입니다.
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
* 연산 지연: $Delay_{comp} = \frac{Q_{local} + 1}{2 \cdot ServiceRate} = \frac{Q_{local} + 1}{4.0}$
  - $ServiceRate = 2$: 채널 상태 $Comm$과 무관하게 독립적으로 고정된 단말기 고유 연산 성능 스펙 (이웃 에지 노드의 연산 속도 분모인 $8.0$에 비해 절반 수준의 처리 능력)
* 에너지 소모: $EnergyConsumed = EnergyCost(Comm) \in \{0.8, 0.5, 0.3\}$

#### 2) 이웃 노드 $i \in \{1, 2\}$ 오프로딩 ($a = 1, 2$)
* 전송 지연: $Delay_{trans} = ChannelFactor(Comm) \cdot 1.0$
  - $ChannelFactor(Comm) \in \{1.5, 1.0, 0.5\}$ (채널 품질 상태가 좋음(2)일 때 전송 지연이 가장 작음)
* 연산 지연: $Delay_{comp} = \frac{Q_{ni} + 1}{8.0} + 0.05$ (이웃 에지 노드는 연산 속도 분모 $8.0$을 가진 고성능 프로세서 장착)
* 에너지 소모: $EnergyConsumed = EnergyCost(Comm) \cdot 0.6$ (오프로딩 시 무선 송출에 소모되는 전력은 로컬 연산 대비 $60\%$ 수준으로 절감)

#### 3) 의도적 드롭 ($a = 3$)
* 지연 시간 및 소모 에너지 모두 0으로 산출 ($Delay_{trans} = 0.0, Delay_{comp} = 0.0, EnergyConsumed = 0.0$)

---

### 3.3 대기열 혼잡 패널티 ($Penalty_{queue}$ & $Penalty_{neighbor}$)

버퍼가 포화 상태에 가까워질수록 패널티를 비선형적(제곱 형태)으로 부과하여 학습 불안정성을 예방하고 시스템 붕괴 임계점(Buffer Overflow)을 에이전트가 회피하도록 합니다.

#### 1) 로컬 대기열 패널티 ($Penalty_{queue}$)
$$Penalty_{queue} = \beta_{local} \cdot \left(\frac{Q_{local}}{MaxLocalQueue}\right)^2 \quad (\text{단, } MaxLocalQueue = 5.0)$$
* *Standard 설정*: $\beta_{local} = 5.0$
* *Cliff / Sparse 설정*: $\beta_{local} = 0.0$ (큐 패널티 미적용)
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

### 3.5 상태 전이 확률 (State Transition Probability, $P(S_{next} \mid S, a)$) 설계 및 구현

시스템의 다음 상태 $S_{next}$로의 전이는 이전 상태 $S$와 에이전트의 행동 $a$에 의해 결정됩니다. 본 MDP 모델은 다음 상태를 구성하는 개별 상태 변수들이 조건부 독립(Conditional Independence)이라는 가정 하에 설계되었습니다.

#### 1) 결합 전이 확률 (Joint Transition Probability)
개별 상태 변수들의 조건부 독립성에 따라, 결합 상태 전이 확률은 각 변수별 전이 확률의 곱으로 다음과 같이 정의됩니다.

$$P(S_{next} \mid S, a) = P(Task\_Type_{next}) \cdot P(Comm\_State_{next} \mid Comm\_State) \cdot P(Local\_Queue_{next} \mid q_{local\_act}, Comm\_State) \cdot P(N1\_Queue_{next} \mid q_{n1\_act}) \cdot P(N2\_Queue_{next} \mid q_{n2\_act})$$

여기서 각 변수의 다음 상태 전이는 현재 시점의 상태 정보와 선택된 행동 $a$에 의해서만 결정됩니다.

#### 2) 상태 인덱스 맵핑 (State Index Encoding)
총 $3,630$개의 이산 상태 공간을 1차원 평탄화 인덱스(Flat Index) $s\_idx$로 변환하기 위해 다음과 같은 규칙을 사용합니다.

$$s\_idx = Task\_Type \times 1815 + Comm\_State \times 605 + Local\_Queue \times 121 + Neighbor\_1\_Queue \times 11 + Neighbor\_2\_Queue$$

이 수식은 각 상태 변수의 크기($|Task\_Type|=2, |Comm\_State|=3, |Local\_Queue|=5, |Neighbor\_1\_Queue|=11, |Neighbor\_2\_Queue|=11$)에 따른 자릿수 가중치를 기반으로 고유한 인덱스를 매핑합니다.

#### 3) 각 상태 변수별 세부 전이 규칙 (Individual Variable Transition Rules)

##### (1) 태스크 종류 (Task\_Type)
신규 태스크의 종류는 현재 상태 및 에이전트의 행동과 무관하게 매 타임스텝마다 독립적으로 생성됩니다. 두 종류의 태스크(URLLC 및 eMBB)가 발생할 확률은 각각 $0.5$로 동일합니다.
$$P(Task\_Type_{next} = nt) = 0.5 \quad \text{for } nt \in \{0, 1\}$$

##### (2) 통신 상태 (Comm\_State)
통신 상태는 1차원 무작위 워크(1D Random Walk) 모델을 따릅니다. 이전 상태 대비 채널의 상태 변화 $\Delta c \in \{-1, 0, 1\}$에 대한 전이 확률은 각각 $[0.1, 0.8, 0.1]$이며, 경계 조건 $\{0, 1, 2\}$를 벗어나지 않도록 클리핑(Clipping)을 수행합니다.
각 상태별 구체적인 전이 확률 식은 다음과 같습니다.

* **$Comm = 0$ (불량)인 경우**:
  $$P(Comm_{next} = 0 \mid Comm = 0) = 0.9$$
  $$P(Comm_{next} = 1 \mid Comm = 0) = 0.1$$
  $$P(Comm_{next} = 2 \mid Comm = 0) = 0.0$$

* **$Comm = 1$ (보통)인 경우**:
  $$P(Comm_{next} = 0 \mid Comm = 1) = 0.1$$
  $$P(Comm_{next} = 1 \mid Comm = 1) = 0.8$$
  $$P(Comm_{next} = 2 \mid Comm = 1) = 0.1$$

* **$Comm = 2$ (좋음)인 경우**:
  $$P(Comm_{next} = 0 \mid Comm = 2) = 0.0$$
  $$P(Comm_{next} = 1 \mid Comm = 2) = 0.1$$
  $$P(Comm_{next} = 2 \mid Comm = 2) = 0.9$$

##### (3) 로컬 대기열 (Local\_Queue)
행동 $a$가 선택되면 우선 로컬 대기열에 태스크가 추가되어 행동 후 대기열 크기 $q_{local\_act}$가 결정됩니다.
* $a = 0$ (로컬 처리)인 경우: $q_{local\_act} = \min(4, q_{local} + 1)$ (단, $q_{local} + 1 \ge 5$ 이면 오버플로우 드롭 발생 및 드롭 플래그 활성화)
* $a \neq 0$인 경우: $q_{local\_act} = q_{local}$

이후 로컬 서비스 처리율 $S_{local} = 2$만큼 태스크가 소모된 후($q_{served}$), 외부로부터 신규 태스크 유입량 $arr \sim \text{Poisson}(\lambda)$이 더해져 다음 상태의 대기열 크기 $q_{local, next}$가 결정되며 물리 한도인 $4$로 클리핑됩니다.
$$q_{served} = \max(0, q_{local\_act} - 2)$$
$$q_{local, next} = \min(4, q_{served} + arr)$$

푸아송 분포의 도착 확률 질량 함수를 $P(arr = k) = \frac{\lambda^k e^{-\lambda}}{k!}$라 할 때, 로컬 대기열의 구체적인 상태 전이 확률 식은 다음과 같습니다.
* $q_{local, next} < 4$ 인 경우 ($q_{local, next} \ge q_{served}$):
  $$P(q_{local, next} \mid q_{local\_act}, Comm\_State) = \begin{cases} \frac{\lambda^{(q_{local, next} - q_{served})} e^{-\lambda}}{(q_{local, next} - q_{served})!} & \text{if } q_{local, next} \ge q_{served} \\ 0 & \text{otherwise} \end{cases}$$
* $q_{local, next} = 4$ 인 경우 (버퍼 포화):
  $$P(q_{local, next} = 4 \mid q_{local\_act}, Comm\_State) = \sum_{k = 4 - q_{served}}^{\infty} \frac{\lambda^k e^{-\lambda}}{k!} = 1 - \sum_{k=0}^{3 - q_{served}} \frac{\lambda^k e^{-\lambda}}{k!}$$

##### (4) 이웃 노드 대기열 (Neighbor\_Queues)
각 이웃 노드 $i \in \{1, 2\}$의 대기열은 행동 $a$에 따라 행동 후 대기열 크기 $q_{ni\_act}$로 업데이트됩니다.
* $a = i$ (이웃 노드 오프로딩)인 경우: $q_{ni\_act} = q_{ni} + 1$ (최대 $11$)
* $a \neq i$인 경우: $q_{ni\_act} = q_{ni}$

그 후 이웃 노드 자체의 확률적 처리 속도 $S_{ni} \in \{1, 2\}$ 만큼 큐 크기가 감소하며, 각각 $0.5$의 확률을 가집니다.
$$P(S_{ni} = 1) = 0.5, \quad P(S_{ni} = 2) = 0.5$$

감소 후의 대기열 크기는 $[0, 10]$ 범위로 클리핑됩니다.
$$q_{ni, next} = \min(10, \max(0, q_{ni\_act} - S_{ni}))$$

구체적인 전이 확률 규칙은 다음과 같이 적용됩니다.
* $q_{ni\_act} \ge 2$ 인 경우:
  $$P(q_{ni, next} = q_{ni\_act} - 1 \mid q_{ni\_act}) = 0.5$$
  $$P(q_{ni, next} = q_{ni\_act} - 2 \mid q_{ni\_act}) = 0.5$$
* $q_{ni\_act} = 1$ 인 경우:
  $$P(q_{ni, next} = 0 \mid q_{ni\_act} = 1) = 1.0$$
* $q_{ni\_act} = 0$ 인 경우:
  $$P(q_{ni, next} = 0 \mid q_{ni\_act} = 0) = 1.0$$

(예: `trans_n[q_curr, q_next] += 0.5`를 두 차례 적용하여, $S_{ni}$의 값에 따른 전이 확률을 합산합니다.)

---

### 3.6 푸아송 분포(Poisson Distribution) 활용 및 에피소드(Episode) 생성 방식

네트워크 환경의 무작위 태스크 도착 현상을 모사하고, 강화학습 에이전트가 넓은 상태 공간을 효율적으로 탐색할 수 있도록 다음과 같이 수식적·알고리즘적 설계를 적용하였습니다.

#### 1) 푸아송 분포 (Poisson Distribution) 활용
단위 시간당 도착하는 신규 태스크의 개수 $k$는 평균 도착률 $\lambda$를 따르는 푸아송 분포의 확률 질량 함수(PMF)에 의해 결정됩니다.
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

* **모델 해석(Model-based calculations - `build_mdp_model.py`)**:
  - 상태 전이 확률 행렬 $P$를 수학적으로 생성하기 위해 푸아송 PMF를 사용합니다. 로컬 처리 이후 남은 큐 크기($Q_{served}$)와 신규 유입량($arr$)의 합이 다음 단계의 로컬 큐 크기 $Q_{next}$로 전이될 확률을 다음과 같이 계산합니다.
    $$P(Q_{next} \mid Q_{served}, \lambda) = \sum_{arr} P(X = arr) \quad (\text{단, } Q_{next} = \min(4, Q_{served} + arr))$$
  - 또한, 버퍼 한도(4)를 넘어서 발생하는 기대 백그라운드 오버플로우 드롭 수($\mathbb{E}[num\_bg\_drops]$)의 수학적 기댓값을 구하는 데 활용됩니다.
    $$\mathbb{E}[num\_bg\_drops] = \sum_{arr \ge 5 - Q_{served}} P(X = arr) \cdot (Q_{served} + arr - 4)$$
* **시뮬레이션 환경 (Model-free Environment - `mdc_mdp_env.py`)**:
  - 실제 스텝마다 푸아송 분포로부터 도착 태스크 수를 난수 샘플링합니다. 연산 속도 향상을 위해 초기화 시점에 `np.random.poisson`을 사용하여 100,000개의 유입량 샘플을 버퍼로 사전 생성한 후 순환 참조하여 적용합니다.
    $$arrival = \text{Poisson}(\lambda)$$

#### 2) 에피소드 (Episode) 생성 및 진행 방식
학습 및 평가 단계에서 에이전트의 상태 전이 경험을 생성하기 위해 에피소드를 다음과 같이 구성합니다.

* **최대 스텝 수 (Max Steps)**: 에피소드당 최대 스텝 수는 **$1,000$ 스텝**으로 제한됩니다. 즉, $t=1,000$에 도달하면 `truncation = True`가 반환되어 에피소드가 강제로 종료됩니다.
* **학습 시 - 무작위 시작 (Exploring Starts - `random_start=True`)**:
  - 학습 시에는 에이전트가 상태 공간 전체를 고르게 방문할 수 있도록 매 에피소드 시작 시 상태 변수들을 무작위로 초기화합니다.
    $$Q_{local} \sim \text{Uniform}(0, 4), \quad Q_{n1}, Q_{n2} \sim \text{Uniform}(0, 10), \quad Comm \sim \text{Uniform}(0, 2), \quad Task \sim \text{Uniform}(0, 1)$$
  - 이를 통해 특정 안전한 상태에만 고착(Local Minima)되는 문제를 방지하고, 벼랑 끝 상태나 큐가 가득 찬 극한 상태에서의 복원 대책을 강건하게 학습합니다.
* **평가 시 - 고정 시작 (`random_start=False`)**:
  - 학습이 완료된 알고리즘의 절대 성능을 일관성 있게 검증하기 위해, 평가 모드에서는 항상 동일한 시작 상태에서 출발합니다.
    $$S_{start} = [Task=0, Comm=1, Q_{local}=0, Q_{n1}=0, Q_{n2}=0]$$
  - 즉, URLLC 태스크가 들어오고 채널 품질이 보통이며 모든 대기열이 비어있는 정형화된 상태에서 시뮬레이션을 개시하여 500회 평가 에피소드의 누적 보상 평균을 측정합니다.

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
* **수정**: **20,000 에피소드** 학습으로 연장 및 감쇠 스케줄러 완화 (총 에피소드의 **$60\%$** 시점인 $episodes \times 0.6$ 스텝까지 서서히 감쇠)
  - 탐험 감쇠는 `decay = np.exp(np.log(eps_min)/(episodes * 0.6))` 공식을 활용하여 탐험율 $\epsilon$이 $20,000 \times 0.6 = 12,000$ 에피소드 시점까지 점진적으로 감소하도록 설계되었습니다.
  - 학습 수렴 횟수를 4배 늘려 희소 상태 공간(State Sparsity)의 방문 빈도를 확보함.
  - 탐험 확률 $\epsilon$이 더 오랜 기간 동안 큰 값을 유지하도록 스케줄링을 완화하여, 에이전트가 벼랑 끝 상태나 드롭 상태의 정확한 패널티 기대치를 충분히 업데이트할 시간을 확보함.

이 수정을 통해 Q-Learning과 Expected SARSA 에이전트는 미방문 상태에서의 오작동(비어 있는 대기열에서의 태스크 드롭)을 근절하고, 수학적 상한선인 DP(동적 계획법) 정책의 최적 보상 수준과 드롭 성능에 완벽하게 수렴하였습니다.

---

## 5. 코드베이스 검증 기록 (Verification Note)

* **최근 검증 일시**: 2026-05-24
* **검증 대상 코드**: `mdc_mdp_env.py` (비기능적 주석 수정 적용됨) 및 `build_mdp_model.py`
* **검증 내용**: 
  - 단말기 및 이웃 노드의 계산/전송 지연 산출식 분모와 가중치들의 정밀성 검증 완료.
  - Cliff 설정의 보상 함수에서 로컬 큐 패널티($Penalty_{queue}$)가 수식에 실제로 합산되지 않음을 확인하여, 이에 맞게 공식 사양을 $\beta_{local} = 0.0$(큐 패널티 미적용)으로 정밀 수정하여 코드와 문서의 싱크를 100% 일치시킴.
  - 비기능적 코드 변경(주석 추가 등)이 환경의 수치적 전이나 보상 계산 흐름에 영향을 주지 않음을 물리적 검증 완료함.

