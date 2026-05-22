# Update Log (변경 이력)

본 문서는 프로젝트(코드/설계/문서)의 **누적 변경 이력**을 기록합니다.

## 작성 규칙
- 메인 문서는 [`README.md`](./README.md)이며, 본 문서는 README를 보완하는 changelog입니다.
- 신규 변경은 **이 파일 가장 위(최신)** 에 추가합니다 (역연대순, 최근 항목이 위).
- 각 변경 그룹은 `# YYYY-MM-DD — 한 줄 요약` 헤더로 시작합니다.
- 항목 내부는 다음 구조를 권장합니다.
  - 🔴 문제 / 동기
  - ✅ 수정 내용 (코드 스니펫 포함)
  - 🎯 효과 / 영향 파일
- 결과물(`models/*.pkl`, `q_table_*.npy`, `results/`) 재생성이 필요한 경우 항목 말미에 명시합니다.
- README의 명세와 충돌 시 **본 문서의 최신 항목이 우선**입니다. 안정화된 변경은 추후 README로 반영합니다.

---

# 2025-05-22 — MDP 설계 1차 개정 (rubric 대응)

PDF rubric(`60375757-EL5001_RL_Proj_01...pdf`)의 **"Formulate real-world problem as MDP"** 및 **"Discussion & Justification"** 항목에서 감점 위험이 있던 설계 결함을 정리하고, `mdc_mdp_env.py`에 반영한 1차 개정 내용입니다.

수정 순서: **#1 → #2 → #5 → #3 → #4 (+ #7 동반 수정)**

---

## 개정 요약 (TL;DR)

| # | 항목 | 수정 전 | 수정 후 | 영향 파일 |
|---|---|---|---|---|
| 1 | Neighbor overflow drop | full queue에 offload 시 무페널티로 task 소멸 | `q_n ≥ 10`이면 `is_dropped=True`, 전송 비용도 부과하지 않음 | `mdc_mdp_env.py` |
| 2 | Neighbor queue penalty | local queue에만 quadratic penalty | local + N1 + N2 모두에 quadratic penalty (β_n=3.0) | `mdc_mdp_env.py` |
| 5 | Comm_State 의미 혼선 | "통신 상태"라는 이름인데 local CPU 속도까지 결정 | **Sys_State**(채널+CPU 종합 상태)로 명명 일원화, docstring·라벨 갱신 | `mdc_mdp_env.py`, `build_mdp_model.py`, `visualize_mdp_results.py` |
| 3 | Neighbor background load 부재 | 에이전트가 offload 안 하면 N1/N2 큐는 줄기만 함 | 매 step `Poisson(0.7)` 외부 사용자 도착을 N1/N2에 주입 | `mdc_mdp_env.py` |
| 4 | Positive success reward 부재 | 모든 보상이 음수 (cost-min) | 정상 처리 시 `+w_task · 1.0` (standard 보상만) | `mdc_mdp_env.py` |
| 7 | Local overflow 이중부과 | local_q=4에서 action=Local → delay+energy 비용도 받고 drop penalty도 받음 | overflow 확정 시 delay/energy를 부과하지 않음 (#1과 동일 패턴) | `mdc_mdp_env.py` |

---

## 1. Neighbor Overflow Drop 처리 (#1)

### 🔴 문제
기존 코드 (`mdc_mdp_env.py`, 수정 전):
```python
elif action == 1 or action == 2: # Offload
    idx = action - 1
    ...
    self.neighbor_qs[idx] += 1   # ← 오버플로우 검사 없음
...
self.neighbor_qs[i] = np.clip(self.neighbor_qs[i], 0, 10)  # 조용히 잘림
```
- `q_n = 10` 상태에서 offload하면 step 끝의 `clip`으로 silently 사라지면서도 `is_dropped` 플래그는 켜지지 않음 → drop penalty 미부과.
- Local action(`q_l ≥ 5`)에서는 overflow drop을 인식하면서 neighbor에서는 인식하지 않는 **비대칭 결함**.
- 검토 의견 보강: `+1` 후 service가 먼저 빠지면 실제로 9 또는 10에 머무를 수도 있지만, **full queue에 task를 받는 것이 무페널티**라는 핵심 문제는 동일.

### ✅ 수정
```python
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
```
- Local overflow와 동일한 처리 정책 적용: **거절(reject) 시 전송 비용 0, drop penalty만**.

### 🎯 효과
- Action 1/2가 더 이상 "공짜 카드"가 아님 → 에이전트가 neighbor 혼잡을 회피해야 할 동기 발생.
- DP에서도 P(s,a,s')와 R(s,a)에 정확히 반영됨 (`build_mdp_model.py`가 sampling 기반이라 자동 반영).

---

## 2. Neighbor Queue Penalty 추가 (#2)

### 🔴 문제
```python
# 수정 전
penalty_queue = self.beta * ((self.local_q / self.max_local_q) ** 2)
```
- README는 "큐가 찰수록 점진적으로 페널티 강화"를 자랑하지만, 실제로는 **local_q 한 개**에만 적용.
- Neighbor 큐 길이는 `delay_comp = (q_n+1)/4 + 0.05`로 간접적으로만 보상에 반영 → offloading 편향.

### ✅ 수정
새 하이퍼파라미터 `beta_n = 3.0`, `max_neighbor_q = 10.0` 도입 후:
```python
penalty_queue = (
    self.beta   * (self.local_q       / self.max_local_q)    ** 2
  + self.beta_n * (self.neighbor_qs[0] / self.max_neighbor_q) ** 2
  + self.beta_n * (self.neighbor_qs[1] / self.max_neighbor_q) ** 2
)
```
- 세 큐 모두 동일한 quadratic 형태로 페널티.
- `β_n < β`로 둔 이유: neighbor는 max 용량이 10으로 단말(5)보다 크고, 본래 edge server가 더 큰 버퍼를 가짐.
- 적용 범위: **standard 보상만**. cliff/sparse는 의도적 미니멀 신호이므로 변경하지 않음.

### 🎯 효과
- "Local만 가득 차면 안 좋고 neighbor는 가득 차도 괜찮다"는 비대칭 학습 신호 제거.
- 세 큐의 동시 혼잡을 시스템 전체 비용으로 정의 → 부하 분산 정책이 자연스럽게 학습됨.

---

## 5. Comm_State → Sys_State 명명 일원화 (#5)

### 🔴 문제
```python
self.service_rates    = [1, 2, 3]      # Local CPU speed
self.channel_factors  = [1.5, 1.0, 0.5]
self.energy_costs     = [0.8, 0.5, 0.3]
...
s_rate = self.service_rates[comm]   # ← comm으로 로컬 CPU 인덱싱
```
- 변수명은 "통신 상태"인데 **로컬 CPU service rate까지** 동일 인덱스가 결정.
- Rubric의 "Is the state sufficient / what was excluded?"에서 변수 의미가 깨졌다고 지적당할 위험.

### ✅ 수정
변수와 라벨을 **Sys_State**(System State, 채널 품질 + CPU 컨텐션 결합 상태)로 통일.

- `mdc_mdp_env.py`:
  - `self.comm_state` → `self.sys_state`
  - step() 내부 unpack: `task_type, sys_state, q_local, q_n1, q_n2 = self.state`
  - 클래스 docstring에 "joint index that captures both channel quality and local CPU contention" 명시
  - 주석: `# Performance parameters (indexed by sys_state: 0:Poor, 1:Normal, 2:Good)`
- `build_mdp_model.py`:
  - `get_comm_probs` → `get_sys_state_probs`
  - `env.comm_state = comm` → `env.sys_state = sys_state`
- `visualize_mdp_results.py`:
  - heatmap ylabel: `"Comm State"` → `"Sys State"`

### 🎯 효과
- "왜 통신 상태가 CPU 속도를 결정하나?" 질문에 대해 **"채널 혼잡과 단말 자원 컨텐션은 모바일 환경에서 상관관계가 강해 한 변수로 결합 모델링"** 이라고 정당화 가능.
- 상태공간은 그대로 3,630개 유지 (DP 비용 변화 없음).

---

## 3. Neighbor Background Load 추가 (#3)

### 🔴 문제
```python
# 수정 전: neighbor 큐는 오직 service로만 감소
self.neighbor_qs[i] = max(0, self.neighbor_qs[i] - np.random.randint(1, 3))
```
- 에이전트가 offload를 안 하면 N1, N2는 항상 0으로 수렴.
- 실제 edge server는 **다른 사용자의 task도 동시에 처리**해야 하므로 비현실적.
- "왜 두 neighbor 중 하나를 선택해야 하는가?"라는 핵심 의사결정 자체가 약해짐.

### ✅ 수정
새 하이퍼파라미터 `neighbor_bg_lambda = 0.7` 도입:
```python
for i in range(2):
    self.neighbor_qs[i] = max(0, self.neighbor_qs[i] - np.random.randint(1, 3))
    # Background arrivals from other users
    bg_arrival = np.random.poisson(self.neighbor_bg_lambda)
    self.neighbor_qs[i] = np.clip(self.neighbor_qs[i] + bg_arrival, 0, 10)
```
- N1, N2 각각에 매 step 독립적인 Poisson 도착.
- λ=0.7은 service rate 평균 1.5보다 작아 무한 발산하지 않음 (정상 상태 보장).

### 🎯 효과
- Neighbor 큐가 시간에 따라 변동 → "지금은 N1이 한가하지만 다음엔 N2가 더 비어 있을 수 있음" 같은 dynamic load-balancing 의사결정이 의미를 가짐.
- Sanity check 결과: idle 50 step 동안 N1 max=4, mean=0.78로 적절한 변동 확인.

---

## 4. Positive Success Reward 도입 (#4)

### 🔴 문제
- 모든 보상이 ≤ 0 (drop=−20, 처리=−cost).
- 학습 목표가 "손실 최소화"뿐이라 **eMBB(throughput)의 가치**가 정량화되지 않음.
- 극단적으로 학습 초기 임의 정책에서 "전부 drop" 전략이 cost=−20.0 vs 처리 시도 cost=−4.7 같은 비교에서 drop이 손해여도, 정상 처리에 **양의 신호가 전혀 없어** trade-off 해석이 어려움.

### ✅ 수정
새 하이퍼파라미터 `success_bonus = 1.0` 도입, standard 보상에만 추가:
```python
success_term = (w_task * self.success_bonus) if is_success else 0.0
reward = -(cost_delay_energy + penalty_queue + penalty_drop) + success_term
```
- URLLC(`w_task=2.0`)는 보너스 +2.0, eMBB(`w_task=0.5`)는 +0.5.
- 크기가 작아 cost/penalty 신호를 덮어쓰지 않음.

### 🎯 효과
- 정상 처리가 "0이 아닌 양의 가치"를 가지므로 throughput-oriented 정책이 표현 가능.
- 검토 의견대로 **이건 버그가 아니라 목적함수 설계 강화**이므로, 보고서에서는 "throughput을 명시적으로 보상해 SLA 충족 정책을 유도"로 정당화 권장.

---

## 7. Local Overflow 이중부과 제거 (동반 수정)

### 🔴 문제
```python
# 수정 전: action=Local일 때 항상 delay/energy 계산 후 overflow 검사
if action == 0:
    delay_comp = (self.local_q + 1) / (s_rate * 2.0)
    energy_consumed = self.energy_costs[comm]
    self.local_q += 1
...
if self.local_q >= 5:
    is_dropped = True
    self.local_q = 4
```
- local_q=4에서 action=Local → 처리 비용(delay+energy)을 받고도 drop penalty까지 받음 (**이중 카운팅**).
- Action 0의 overflow drop과 Action 3의 intentional drop이 같은 결과를 만들면서 보상은 다름.

### ✅ 수정
overflow를 **선제 검사**해서, 거절 시 비용을 발생시키지 않음:
```python
if action == 0:
    if self.local_q >= 4:
        is_dropped = True   # no delay/energy charged
    else:
        s_rate = self.service_rates[sys_state]
        delay_comp = (self.local_q + 1) / (s_rate * 2.0)
        energy_consumed = self.energy_costs[sys_state]
        self.local_q += 1
        is_success = True
```
- Neighbor overflow(#1)와 **동일한 패턴**으로 일관성 확보.

### 🎯 효과
- 보상 함수의 의미가 명확: "처리에 성공하면 cost 부과, 거절되면 drop penalty"로 깔끔하게 분리.
- Intentional drop(action=3) vs overflow drop(action=0, q=4) 차이가 사라짐 → 에이전트는 "어차피 drop될 거면 action=3으로 의도적으로 drop"하는 자연스러운 선택을 학습.

---

## 새로 도입된 하이퍼파라미터

| 이름 | 값 | 의미 |
|---|---|---|
| `beta_n` | 3.0 | per-neighbor queue penalty scale (#2) |
| `max_neighbor_q` | 10.0 | neighbor queue 정규화 분모 (#2) |
| `neighbor_bg_lambda` | 0.7 | 외부 사용자 background arrival rate (#3) |
| `success_bonus` | 1.0 | 처리 성공 시 base 보너스 (#4, standard only) |

### Ablation 권장 (보고서용)

Rubric의 "ablation study" 요구에 대응하기 위해 다음 변형 실험을 추천:

| 실험 | 변경 | 기대 관찰 |
|---|---|---|
| no neighbor penalty | `beta_n = 0` | 무분별 offload, neighbor 큐 포화 빈도 ↑ |
| no background load | `neighbor_bg_lambda = 0` | DP 정책이 neighbor 선택을 거의 무차별로 학습 |
| no success bonus | `success_bonus = 0` | eMBB 정책이 drop 편향, throughput 지표 ↓ |
| 큰 drop penalty | `gamma = 100` | local_q=4 회피 강화, 의도적 drop 빈도 ↓ |

---

## 변경된 파일

| 파일 | 변경 내용 |
|---|---|
| `mdc_mdp_env.py` | 전체 step() 로직 재정의 (#1, #2, #3, #4, #5, #7), 하이퍼파라미터 4종 추가, docstring 보강 |
| `build_mdp_model.py` | `comm_state` → `sys_state` 명명 일원화, 함수명 `get_comm_probs` → `get_sys_state_probs` |
| `visualize_mdp_results.py` | heatmap 라벨 "Comm State" → "Sys State" |

기존 학습 결과(`models/*.pkl`, `q_table_*.npy`)는 보상/전이 정의가 바뀌었으므로 **재생성 필요**합니다. `run_mdp_pipeline.sh`를 다시 실행하세요.

---

## 미수정으로 남긴 항목 (정당화로 방어 권장)

검토 의견에서 "버그가 아닌 정당화 부족"으로 분류된 항목들은 코드 수정 대신 **보고서/슬라이드에서 설계 의도를 명시**해 방어합니다.

| # | 항목 | 방어 논리 |
|---|---|---|
| 6 | Task_type i.i.d. 전이 | 현재 task urgency가 reward weight를 통해 action 선택에 영향 → dead weight 아님. 상태공간 단순화 위한 의도적 선택. |
| 8 | step ordering | "action → reward → service → arrival" 순서를 README에 다이어그램으로 명시. |
| 10 | cliff noise | DP는 expected reward로 푸는 반면 SARSA는 sample reward를 받는 차이를 보여주려는 의도된 stress test임을 명시. |
| 11 | sampling 기반 P 추정 | "Strict MDP" 표현을 "수치적으로 정의된 MDP"로 톤다운하고, 2000-sample 표준오차를 보고. |
| 12 | episode 종료 | infinite-horizon discounted MDP를 1000-step truncation으로 근사한다고 명시 (γ=0.95에서 effective horizon ≈ 20 step ≪ 1000). |
| 13 | 큐 내 task type 미반영 | i.i.d. task 도착 + FIFO 가정 하에서 marginal로 잘 근사된다고 정당화. 상태폭발 방지가 명시적 선택. |
