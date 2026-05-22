# MDC MDP Task Offloading Project

이 프로젝트는 MDC(Mobile Distributed Computing) 태스크 오프로딩 문제를 MDP(Markov Decision Process)로 모델링하고, 다양한 알고리즘을 통해 최적의 정책을 학습하고 비교하는 프레임워크입니다.

> 📝 **변경 이력**: 본 README는 프로젝트의 **메인 문서**입니다. 설계/구현/문서가 변경될 때마다 변경 사유와 영향 범위를 [`Update.md`](./Update.md)에 누적 기록합니다. 최신 사양과 README가 충돌하는 경우 `Update.md`의 최신 항목을 우선으로 합니다.

---

## 1. 기존 코드의 문제점 분석 (왜 완전한 MDP가 아니었는가?)

원본 코드는 시스템의 실제 물리적 상태를 은닉(Hidden)하고 에이전트에게 요약된 정보만 전달했기 때문에 MDP가 아닌 POMDP(부분 관측 가능한 MDP)에 가까웠습니다.

- **상태 추상화로 인한 마르코프 성질 위배:** 물리 큐 크기(0~50)를 에어전트에게는 4단계 레벨(0~3)로 압축해 보여주었습니다. 이로 인해 동일한 레벨이더라도 내부 실제 큐 크기에 따라 다음 상태로 갈 확률이 달라져 마르코프 성질이 깨졌습니다.
- **이웃 노드 정보의 블라인드 스팟:** 6개의 이웃 중 하나를 골라 오프로딩할 수 있으면서 상태 공간에는 '가장 상태가 좋은 이웃 1개'의 정보만 있었습니다. 다른 이웃을 선택했을 때의 결과를 현재 상태에서 예측할 수 없었습니다.
- **DP(Dynamic Programming) 적용 불가:** 전이 확률 행렬 $P(s' \mid s, a)$와 기대 보상 함수 $R(s, a)$를 수학적으로 정의할 수 없는 Model-free 시뮬레이션 구조였습니다.

---

## 2. 모델 재설계 전략 (차원의 저주 극복)

수십 조 개의 상태 공간을 DP 연산이 가능한 **수천 개 수준**으로 줄이기 위해 다음과 같은 통폐합 및 다이어트를 진행했습니다.

- **행동 공간의 이진화 및 구체화:** 이웃 6개를 모두 고르는 대신, 신호가 가장 잘 잡히는 상태가 좋은 이웃 노드 2개(Neighbor 1, 2)로 대상을 좁혔습니다.
- **환경 변수의 통합:** 채널 상태(3)와 대역폭(2)을 각각 추적하지 않고, 전송 속도와 단말 CPU 컨텐션을 결합한 `Sys_State` (3단계) 변수 하나로 통합했습니다.
- **큐 크기의 마르코프 상태화:** 압축된 레벨을 버리고, **실제 물리 큐 크기의 정수 값 자체를 상태 공간에 직접 편입**시켜 은닉 상태를 완전히 제거했습니다.

---

## 3. 큐(Queue) 크기 설정 및 개선

|**구분**|**기존 설정**|**최종 개선 설정**|**현실적인 이유**|
|---|---|---|---|
|**로컬 큐 (Local Q)**|0 ~ 15 (16단계)|**0 ~ 4 (5단계)**|단말기는 자원이 부족하므로 즉시 드롭하거나 넘겨야 함|
|**이웃 큐 (Neighbor Q)**|0 ~ 5 (6단계)|**0 ~ 10 (11단계)**|에지 서버는 대용량 처리가 가능하므로 더 큰 버퍼 수용 가능|

---

## 4. 최종 정의된 완전한(Strict) MDP 모델 Specification

### ① 행동 공간 (Action Space) : 4차원
- `0`: 로컬 처리 (Local Processing)
- `1`: 이웃 노드 1로 오프로딩 (Offloading to Neighbor 1)
- `2`: 이웃 노드 2로 오프로딩 (Offloading to Neighbor 2)
- `3`: 의도적 태스크 드롭 (Intentional Drop)

### ② 상태 공간 (State Space) 구성 변수
1. `Task_Type` : 2단계 (0: URLLC, 1: eMBB)
2. `Sys_State` : 3단계 (0: 불량, 1: 보통, 2: 좋음) — 채널 품질과 단말 CPU 컨텐션을 결합한 종합 상태
3. `Local_Queue` : 5단계 (0 ~ 4 정수)
4. `Neighbor_1_Queue` : 11단계 (0 ~ 10 정수)
5. `Neighbor_2_Queue` : 11단계 (0 ~ 10 정수)

**총 상태 공간 개수** = $2 \times 3 \times 5 \times 11 \times 11 = 3,630$개

### ③ Episode 종료 (Termination)

본 MDP는 본질적으로 **continuing (non-episodic) MDP** 입니다. 자연스러운 terminal state(예: task 완료, 시스템 영구 정지)가 존재하지 않으며, Gymnasium 관례에 따라 `max_steps=1000`에서 `truncated=True`로 끊고 `terminated`는 향후 fatal failure 조건을 위해 항상 False로 둡니다.

- 할인율 $\gamma = 0.95$ 하에서 effective planning horizon은 $1/(1-\gamma) = 20$ step.
- 1,000-step rollout은 **약 50배의 effective horizon**을 포함하므로, truncation으로 인한 discounted return 편향은 $\gamma^{1000} \approx 5 \times 10^{-23}$ 수준으로 무시할 수 있습니다.
- 따라서 학습된 정책은 사실상 infinite-horizon discounted 정책에 수렴합니다.

### ④ 상태 공간의 한계 (Approximation Note)

큐는 **길이(count)** 만 저장하며, 큐 내부의 URLLC/eMBB **구성 비율**은 상태에 포함되지 않습니다. 슬롯별 task type을 인코딩하면 상태 공간이 약 3,630 → 3.6M으로 폭증해 tabular DP/RL이 불가능해지기 때문입니다.

본 프로젝트는 (a) task type이 URLLC/eMBB 균등 분포(0.5/0.5)로 도착하고, (b) 보상이 `w_task`에 선형이라는 두 조건 하에서 **expected reward가 큐 내부 composition에 대해 marginalize되는 근사**로 정당화합니다. Function approximation 기반 확장은 future work로 남깁니다.

### ⑤ 재현성 (Reproducibility)

환경 내부의 모든 확률 전이는 Gymnasium이 관리하는 `self.np_random`(seeded Generator)을 사용합니다. `env.reset(seed=...)` 한 번으로 episode 단위 결정성이 확보되며, 학습/평가 스크립트(`train_all_mdp.py --seed 42`)는 episode-indexed seed로 전 파이프라인을 비트 단위 재현 가능합니다.

---

## 5. 정제된 보상 함수 (Refined Reward Function)

학습 안정성과 현실적인 제어 가능성을 극대화하기 위해 다음과 같은 보상 구조를 적용했습니다.

### ① 보상 함수 수식 (Reward Formula)
$$R = -(Cost_{delay\_energy} + Penalty_{queue} + Penalty_{drop})$$

1. **통합 비용 ($Cost_{delay\_energy}$)**: $w_{task} \cdot w \cdot NormDelay + (1-w) \cdot NormEnergy$
   - $w$: 지연 시간과 에너지 간의 가중치 (0.6)
   - $NormDelay, NormEnergy$: 0.1~1.0 사이 정규화된 값
2. **대기열 페널티 ($Penalty_{queue}$)**: $\beta \cdot (Queue/MaxQueue)^2$
   - 제곱 함수(Quadratic)를 사용하여 큐가 찰수록 점진적으로 페널티 강화
3. **드랍 페널티 ($Penalty_{drop}$)**: $\gamma$ (태스크 드랍 시 고정 페널티 20.0 부여)

### ② 개선점 및 특징
- **Normalization**: 서로 다른 물리적 단위(초, 줄)를 정규화하여 특정 지표에 편향되지 않는 균형 잡힌 학습 유도.
- **Trade-off 조절**: 가중치 $w$를 통해 운영 정책에 따른 지연/에너지 민감도 조절 가능.
- **안정적 수렴**: 비선형 페널티 설계를 통해 에이전트가 시스템 붕괴 임계점을 지능적으로 회피하도록 유도.

---

## 6. 주요 파이썬 스크립트 및 역할

### 🟢 `mdc_mdp_env.py`
- **역할**: 3,630개의 이산 상태와 정제된 보상 함수가 구현된 Gymnasium 기반 MDP 환경입니다.

### 🔵 `build_mdp_model.py`
- **역할**: 모든 상태-액션 쌍에 대한 전이 확률($P$)과 기대 보상($R$)을 계산하여 `.pkl` 파일로 저장합니다.

### 🔴 `train_all_mdp.py`
- **역할**: 생성된 모델을 바탕으로 DP(Value/Policy Iteration) 및 RL(SARSA, Q-Learning) 알고리즘을 학습하고 성능을 비교합니다.

### 📊 `visualize_mdp_results.py`
- **역할**: 학습 곡선, 성능 지표 비교 바 차트, DP 최적 정책 히트맵 등을 시각화합니다.
