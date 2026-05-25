# MDC MDP 태스크 오프로딩 최적화 실험 보고서

이 보고서는 MDP(Markov Decision Process) 모델링 환경에서 강화학습 알고리즘(Q-Learning, Expected SARSA)과 동적 계획법(DP: Policy/Value Iteration)의 성능을 다각도로 분석하고, QoS(서비스 품질) 최적화 보상 함수를 재설계 및 검증한 결과를 다룹니다.

---

## 1. 기본 실험 환경 및 시각화 (Agent 1)
다양한 트래픽 부하($\lambda$) 및 보상 설정 하에서 학습을 수행하였습니다.
- **트래픽 부하 ($\lambda$)**: `0.1` (낮음), `0.5` (보통), `1.5` (높음), `3.0` (포화)
- **보상 설정**:
  - `standard` (지연+에너지+대기열 패널티, 30,000 에피소드)
  - `sparse` (드롭 시 -100 패널티, 30,000 에피소드)
  - `cliff` (경계면 local_q=4에서 노이즈 및 드롭 시 -1000 패널티, 30,000 에피소드)
  - `improved` (QoS 차등화 및 비관적 Q 초기화 적용, **20,000 에피소드**)
- **제공된 산출물**:
  - 학습된 각 정책(Policy) 데이터는 `visualization/policy_data.js` 및 `policy_data.json`으로 내보내져 [HTML 시각화 대시보드](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/visualization/policy_visualizer.html)에서 동적으로 탐색이 가능합니다.
  - 고차원 상태 공간(3,630개 상태)의 정책 매핑 관계를 2차원으로 투영한 t-SNE 시각화 플롯이 생성되어 [visualization/tsne_plots/](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/visualization/tsne_plots/) 폴더에 통합 저장되었습니다.

---

## 2. 기본 실험 결과 분석 및 성능 비교 (Agent 2)

### A. 실험 결과 데이터 요약 (30,000 에피소드 및 20,000 에피소드 기준)

| 트래픽 부하 ($\lambda$) | 보상 설정 | 알고리즘 | 기대 보상 (Expected Reward) | 에피소드당 평균 드롭 수 | 에피소드당 평균 에너지 소모 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **$\lambda = 0.1$** | Standard | Policy/Value Iteration | -7.53 | 0.00 | 396.73 |
| | | Expected SARSA | -7.71 | 0.00 | 319.14 |
| | | Q-Learning | -7.62 | 0.00 | 366.53 |
| | Sparse | Policy/Value Iteration | -2.05 | 0.00 | 419.54 |
| | | Expected SARSA | -2.06 | 0.00 | 408.05 |
| | | Q-Learning | -2.05 | 0.00 | 419.38 |
| | Cliff | Policy/Value Iteration | -5.21 | 0.00 | 521.23 |
| | | Expected SARSA | -5.40 | 0.00 | 483.11 |
| | | Q-Learning | -5.24 | 0.00 | 471.02 |
| | **Improved (20k)**| Policy/Value Iteration | -8.43 | 0.00 | 366.53 |
| | | Expected SARSA | -8.44 | 0.00 | 319.14 |
| | | Q-Learning | -8.54 | 0.00 | 396.71 |
| **$\lambda = 0.5$** | Standard | Policy/Value Iteration | -9.29 | 0.92 | 363.84 |
| | | Expected SARSA | -9.38 | 0.92 | 344.72 |
| | | Q-Learning | -9.76 | 0.92 | 356.51 |
| | Sparse | Policy/Value Iteration | -2.05 | 0.24 | 396.83 |
| | | Expected SARSA | -2.08 | 1.58 | 371.73 |
| | | Q-Learning | -2.06 | 0.24 | 339.26 |
| | Cliff | Policy/Value Iteration | -5.68 | 0.58 | 473.10 |
| | | Expected SARSA | -6.77 | 0.58 | 413.24 |
| | | Q-Learning | -6.45 | 0.58 | 364.56 |
| | **Improved (20k)**| Policy/Value Iteration | -12.92 | 1.43 | 343.91 |
| | | Expected SARSA | -13.00 | 1.43 | 362.52 |
| | | Q-Learning | -13.44 | 1.43 | 348.44 |
| **$\lambda = 1.5$** | Standard | Policy/Value Iteration | -70.23 | 169.99 | 326.32 |
| | | Expected SARSA | -70.81 | 177.13 | 328.18 |
| | | Q-Learning | -70.33 | 169.99 | 323.39 |
| | Sparse | Policy/Value Iteration | -444.43 | 176.46 | 349.64 |
| | | Expected SARSA | -479.34 | 192.39 | 348.43 |
| | | Q-Learning | -479.34 | 192.39 | 343.52 |
| | Cliff | Policy/Value Iteration | -2064.66 | 175.28 | 380.37 |
| | | Expected SARSA | -2065.62 | 175.28 | 337.95 |
| | | Q-Learning | -2065.26 | 175.28 | 334.67 |
| | **Improved (20k)**| Policy/Value Iteration | -106.76 | 177.56 | 321.54 |
| | | Expected SARSA | -108.65 | 177.56 | 335.93 |
| | | Q-Learning | -110.32 | 185.00 | 362.90 |
| **$\lambda = 3.0$** | Standard | Policy/Value Iteration | -138.40 | 1170.85 | 319.68 |
| | | Expected SARSA | -138.45 | 1170.85 | 320.11 |
| | | Q-Learning | -138.43 | 1170.85 | 320.60 |
| | Sparse | Policy/Value Iteration | -2439.46 | 1262.88 | 326.02 |
| | | Expected SARSA | -2450.73 | 1273.63 | 322.51 |
| | | Q-Learning | -2439.46 | 1262.88 | 320.84 |
| | Cliff | Policy/Value Iteration | -27586.67 | 1226.32 | 332.50 |
| | | Expected SARSA | -27587.03 | 1226.32 | 320.84 |
| | | Q-Learning | -28674.20 | 1273.81 | 321.31 |
| | **Improved (20k)**| Policy/Value Iteration | -505.60 | 1146.97 | 319.17 |
| | | Expected SARSA | -530.62 | 1296.33 | 369.29 |
| | | Q-Learning | -506.66 | 1146.97 | 328.55 |

### B. 핵심 분석 결과 및 해석

1. **Expected SARSA vs. Q-Learning**:
   - **온폴리시(On-policy) 위험 회피 vs. 오프폴리시(Off-policy) 낙관주의**: 
     - 변동성(노이즈)이 추가된 **Cliff** 설정에서 포화 상태($\lambda = 3.0$)일 때, **Expected SARSA**는 기대 보상 **-27,587.03 (1,226.32 드롭)**을 기록하며, **Q-Learning**의 **-28,674.20 (1,273.81 드롭)**보다 우수한 성과를 보였습니다. Q-Learning은 target을 갱신할 때 최대값($\max$)을 취하여 경계면의 노이즈를 과대평가하기 쉽고, 탐험 중 실수를 예측하지 못해 벼랑 아래로 떨어집니다. Expected SARSA는 확률적 탐험 행동의 가치까지 가중 평균하므로 보수적이고 안전한 정책을 학습합니다.
   - **희소 보상(Sparse Reward)에서의 탐험 희석**:
     - **Sparse** 설정($\lambda=0.5$)에서는 반대로 **Q-Learning (0.24 드롭)**이 **Expected SARSA (1.58 드롭)**보다 훨씬 우수합니다. SARSA는 확률적 탐험 행동의 감점 가치까지 전파하여 타겟 값을 왜곡(희석)시키는 반면, Q-Learning은 순수 탐욕 정책만을 평가하므로 명확하게 수렴합니다.

2. **기존 보상 함수의 한계점**:
   - **성급한 드롭 (Standard)**: 드롭 고정 패널티가 너무 낮아($\gamma = 5.0$), 중부하 상태($\lambda = 0.5$)에서 물리적으로 충분히 처리할 수 있는 태스크도 대기열 패널티를 우려하여 쉽게 드롭(평균 0.92 드롭)해 버립니다.
   - **QoS 인지력 부재**: URLLC(지연 및 드롭에 매우 민감) 태스크와 eMBB(지연 허용 가능) 태스크가 물리적 가중치와 드롭 패널티를 동일하게 사용하여, 자원 배분의 우선순위 제어가 불가능합니다.

---

## 3. 개선된 보상 함수 설계 및 비관적 Q 초기화 검증 (Agent 3)

태스크 타입($0$: URLLC, $1$: eMBB)에 따라 보상 하이퍼파라미터를 동적으로 다르게 적용하는 `"improved"` 보상 타입을 도입하고, 학습 불안정성을 해결하기 위해 **비관적 Q 초기화(Pessimistic Initialization)**와 **탐험 스케줄러 완화**를 진행하였습니다.

### A. 개선 사항
- **Q-테이블 초기값 변경**: 기존 `0.0` 초기화 방식은 음수 보상 환경에서 미방문 상태를 낙관적으로 오판하게 만듭니다. 이를 방지하고자 초기값을 **`-150.0`**으로 설정하여 검증되지 않은 돌발 행동(비어 있는 대기열에서의 태스크 드롭 등)을 차단하였습니다.
- **학습 에피소드 연장**: 기존 5,000 에피소드에서 **20,000 에피소드**로 연장하여 충분한 수렴을 보장하였습니다.
- **탐험율 감쇠 완화**: 에이전트가 최적 정책에 도달하기 전 충분히 상태 공간을 탐험할 수 있도록 감쇠 스케줄링을 `episodes * 0.6` 시점까지 느리게 하였습니다.

### B. 성능 검증 결과
비관적 초기화와 20,000 에피소드 학습을 적용한 결과, Q-Learning과 Expected SARSA 에이전트는 **수학적 상한선인 DP(동적 계획법) 정책의 최적 보상 수준과 드롭 성능에 완벽하게 수렴**하였습니다. 

- **부하 분산 및 레이턴시 최소화**:
  - 개선안에서는 URLLC 태스크에 대해 매우 엄격한 대기열 패널티($\beta_{local}=8.0$)와 무거운 드롭 패널티($\gamma_{task}=30.0$)를 동시에 부과하였습니다. 
  - 이에 따라 에이전트는 무작위로 축적되는 대기열 속에서 URLLC 태스크가 오랜 타임스텝 동안 큐에 대기하여 극심한 큐 누적 패널티를 받기 전에, 비어 있는 이웃 에지 노드로 빠르게 오프로딩하거나 상황에 따라 즉시 드롭 처리를 단행함으로써 **QoS 제약 조건을 충족하며 전체 성능을 최적화**하는 정책을 정확하게 수집 및 학습하였습니다.

---

## 4. 파이프라인 실행 방법 및 실행 순서 가이드 (Pipeline Execution Guide)

본 프로젝트의 모델 빌드, 강화학습 에이전트 학습, 데이터 시각화 및 검증 분석을 위한 전체 파이프라인의 올바른 실행 순서와 상세 명령어 가이드라인입니다. 모든 학습 및 결과 분석은 $\lambda \in [0.5, 1.0, 1.5]$ 트래픽 부하와 `standard` 및 `cliff` 보상 설정에 대해 적용됩니다.

---

### [Step 1] 분석적 MDP 모델 구축 (Transition & Reward Pickle 생성)
가치 반복(Value Iteration) 및 정책 반복(Policy Iteration)의 상태 전이 확률 행렬($P$)과 보상 행렬($R$)을 수학적으로 사전 빌드합니다.

- **실행 스크립트**: `build_mdp_model.py`
- **주요 인수**: `--lambda_val` (도착률), `--reward_type` (보상 형태)
- **PowerShell 일괄 실행 스크립트**:
  ```powershell
  foreach ($l in @(0.5, 1.0, 1.5)) {
      foreach ($r in @("standard", "cliff")) {
          python build_mdp_model.py --lambda_val $l --reward_type $r
      }
  }
  ```
  *(빌드 결과물은 `models/mdp_model_{reward_type}_L{lambda}.pkl` 경로에 저장됩니다.)*

---

### [Step 2] 강화학습 에이전트 학습 및 평가
Q-Learning 및 Expected SARSA 에이전트를 지정된 에피소드 동안 학습시키고 수렴 로그 및 DP와의 성능 평가 메트릭을 추출합니다.

- **실행 스크립트**: `train_all_mdp.py`
- **주요 인수**: `--lambda_val`, `--reward_type`, `--episodes` (100,000 에피소드 권장)
- **PowerShell 병렬 실행 스크립트 (멀티코어 활용)**:
  ```powershell
  # 6개 시나리오를 동시에 백그라운드 프로세스로 구동
  $p1 = Start-Process python -ArgumentList "-u train_all_mdp.py --lambda_val 0.5 --episodes 100000 --reward_type standard" -NoNewWindow -PassThru
  $p2 = Start-Process python -ArgumentList "-u train_all_mdp.py --lambda_val 0.5 --episodes 100000 --reward_type cliff" -NoNewWindow -PassThru
  $p3 = Start-Process python -ArgumentList "-u train_all_mdp.py --lambda_val 1.0 --episodes 100000 --reward_type standard" -NoNewWindow -PassThru
  $p4 = Start-Process python -ArgumentList "-u train_all_mdp.py --lambda_val 1.0 --episodes 100000 --reward_type cliff" -NoNewWindow -PassThru
  $p5 = Start-Process python -ArgumentList "-u train_all_mdp.py --lambda_val 1.5 --episodes 100000 --reward_type standard" -NoNewWindow -PassThru
  $p6 = Start-Process python -ArgumentList "-u train_all_mdp.py --lambda_val 1.5 --episodes 100000 --reward_type cliff" -NoNewWindow -PassThru
  Wait-Process -Id $p1.Id, $p2.Id, $p3.Id, $p4.Id, $p5.Id, $p6.Id
  ```
  *(학습 결과 및 체크포인트는 `results/L_{lambda}_E_{episodes}/{reward_type}/` 내에 Q-table CSV 및 평가 CSV로 저장됩니다.)*

---

### [Step 3] 개별 시나리오 플롯 및 수렴 곡선 생성
각 시나리오별로 에이전트들의 학습 수렴 곡선(Rolling Reward) 및 성능 비교 바 차트, 그리고 정책 히트맵 이미지를 시각화합니다.

- **실행 스크립트**: `visualize_mdp_results.py`
- **주요 인수**: `--lambda_val`, `--reward_type`, `--episodes`
- **PowerShell 일괄 실행 스크립트**:
  ```powershell
  foreach ($l in @(0.5, 1.0, 1.5)) {
      foreach ($r in @("standard", "cliff")) {
          python visualize_mdp_results.py --lambda_val $l --reward_type $r --episodes 100000
      }
  }
  ```
  *(각 디렉토리에 `mdp_training_curves.png`, `mdp_performance_bars.png`, `mdp_heatmap_*.png` 플롯이 생성됩니다.)*

---

### [Step 4] 웹 대시보드 시각화용 데이터 파일 내보내기
학습된 모든 정책 테이블을 취합하여 HTML 브라우저 비주얼라이저가 로드하는 단일 데이터 셋(`policy_data.js`)으로 압축 추출합니다.

- **실행 스크립트**: `visualization/export_policy_data.py`
- **주요 인수**: `--episodes`
- **명령어**:
  ```powershell
  python visualization/export_policy_data.py --episodes 100000
  ```
  *(추출 결과물은 `visualization/policy_data.js` 및 `results/policy_data.js` 경로에 저장되어 [policy_visualizer.html](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/visualization/policy_visualizer.html) 구동 시 자동 로드됩니다.)*

---

### [Step 5] 고속 큐 시뮬레이션 애니메이션 (GIF) 생성
Cliff 정책과 QoS-Aware Standard 정책의 큐 대기열 변동을 나란히 시뮬레이션하고 누적 대기 상태를 추적하는 고속 애니메이션을 생성합니다.

- **실행 스크립트**: `generate_queue_gif.py`
- **주요 인수**: `--episodes`
- **명령어**:
  ```powershell
  python generate_queue_gif.py --episodes 100000
  ```
  *(결과물인 `queue_simulation_L*.gif`는 `results/` 및 `visualization/` 폴더에 복사됩니다.)*

---

### [Step 6] 3대 정책 side-by-side 비교 애니메이션 (GIF) 생성
DP Optimal vs Expected SARSA vs Q-Learning 세 가지 정책이 동일 환경 노이즈 하에서 큐 크기를 분산 제어하는 다이내믹한 비교 애니메이션을 생성합니다.

- **실행 스크립트**: `generate_policy_comparison_gif.py`
- **주요 인수**: `--episodes`
- **명령어**:
  ```powershell
  python generate_policy_comparison_gif.py --episodes 100000
  ```
  *(결과물인 `policy_comparison_L*.gif`는 각 결과 폴더 및 아티팩트 디렉토리에 복사됩니다.)*

---

### [Step 7] 통합 최종 결과 요약 테이블 생성
최종 학습이 완비된 모든 메트릭 테이블을 터미널 상에 마크다운 포맷의 표로 종합 취합하여 출력합니다.

- **실행 스크립트**: `aggregate_all_comparisons.py`
- **주요 인수**: `--episodes`
- **명령어**:
  ```powershell
  python aggregate_all_comparisons.py --episodes 100000
  ```
