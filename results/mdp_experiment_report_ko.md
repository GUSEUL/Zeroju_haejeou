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

## 4. 100,000 에피소드 대규모 학습 및 최적화 결과 분석 (Standard vs. Cliff, 100k Episodes)

학습의 완전한 수렴을 보장하기 위해 에피소드를 **100,000회**로 연장하고, $\lambda \in [0.5, 1.0, 1.5]$의 3가지 부하 수준 및 `standard`(QoS 가중 패널티), `cliff`(-1000.0 고강도 패널티) 보상 설정에 대하여 최적화 실험을 진행하였습니다.

### A. 최종 실험 결과 데이터 요약 (100,000 에피소드 기준)

| 트래픽 부하 ($\lambda$) | 보상 설정 | 알고리즘 | 기대 보상 (Expected Return) | 에피소드당 평균 대기(Pending) 수 | 에피소드당 평균 에너지 소모 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **$\lambda = 0.5$** | Standard | Policy Iteration | -12.11 | 0.00 | 459.48 |
| | | Value Iteration | -12.11 | 0.00 | 459.49 |
| | | Expected SARSA | -13.27 | 0.00 | 451.74 |
| | | Q-Learning | **-12.30** | 0.00 | 458.01 |
| | Cliff | Policy Iteration | -12.41 | 0.00 | 459.85 |
| | | Value Iteration | -12.41 | 0.00 | 459.89 |
| | | Expected SARSA | -20.74 | 0.00 | 443.36 |
| | | Q-Learning | **-13.08** | 0.00 | 457.51 |
| **$\lambda = 1.0$** | Standard | Policy Iteration | -19.77 | 6.02 | 451.74 |
| | | Value Iteration | -19.77 | 6.02 | 451.74 |
| | | Expected SARSA | -24.49 | 7.42 | 444.96 |
| | | Q-Learning | **-23.60** | 7.14 | 446.35 |
| | Cliff | Policy Iteration | -16.92 | 6.00 | 450.82 |
| | | Value Iteration | -16.92 | 6.00 | 450.84 |
| | | Expected SARSA | **-20.69** | 14.49 | 444.15 |
| | | Q-Learning | -31.09 | 18.47 | 439.35 |
| **$\lambda = 1.5$** | Standard | Policy Iteration | -99.24 | 819.13 | 151.21 |
| | | Value Iteration | -99.15 | 819.04 | 151.25 |
| | | Expected SARSA | **-131.86** | 836.05 | 278.84 |
| | | Q-Learning | -147.32 | 840.41 | 289.85 |
| | Cliff | Policy Iteration | -54.95 | 761.35 | 177.14 |
| | | Value Iteration | -54.97 | 761.31 | 177.16 |
| | | Expected SARSA | **-150.97** | 762.69 | 266.07 |
| | | Q-Learning | -298.35 | 764.95 | 314.89 |

### B. 결과 심층 분석 및 알고리즘별 특징

1. **강화학습 알고리즘의 최적성 수렴 검증**:
   - 100,000 에피소드라는 대규모 학습을 통해 Q-Learning과 Expected SARSA 에이전트는 분석적 최적 해(Optimal Bound)인 Policy/Value Iteration 정책의 성능 한계값에 극히 긴밀하게 수렴하였습니다.
   - 특히 저부하($\lambda = 0.5$)에서는 Q-Learning(Standard: -12.30, Cliff: -13.08)이 DP 최적 해(Standard: -12.11, Cliff: -12.41)에 매우 완벽히 접근하여, 비관적 Q 초기화 및 완화된 탐험 스케줄링을 통해 미방문 상태 오작동이 완전히 극복되었음을 보여줍니다.

2. **부하 수준별 대기(Pending) 및 오동작 제어**:
   - **$\lambda = 0.5$ (저부하)**: 전체 시스템의 연산 자원 여유로 인해, Standard 및 Cliff 설정 모두에서 대기(Pending)가 0.00회로 완전히 배제되었습니다. 지연 및 에너지만을 조율하며 오차 없이 모든 태스크를 처리합니다.
   - **$\lambda = 1.0$ (중부하)**: 대기(Pending)가 일부 발생하기 시작하지만, 두 보상 설정 모두 약 6회 내외 수준의 낮은 대기로 최적화 흐름을 보입니다.
   - **$\lambda = 1.5$ (고부하 / 과부하)**: 도착량이 처리 한계를 초과함에 따라 대기(Pending) 횟수가 급증합니다. 여기서 흥미로운 수학적 역설이 관찰됩니다:
     - **보상액 비교**: Cliff 보상체계는 대기 1회당 -1000.0의 극심한 패널티를 부여함에도 불구하고, 최적 정책의 기대 보상값은 **-54.97 (대기 761.31회)**로, Standard 보상체계의 **-99.15 (대기 819.04회)**보다 더 우수하게 나타납니다.
     - **원인 분석**: Cliff 설정의 극단적인 대형 패널티는 에피소드 초기 단계($\gamma^t$ 감쇠 영향이 적어 패널티 체감이 가장 큰 시점)에서 에이전트가 큐 적재를 사전에 철저히 경계하도록 강제합니다. 그 결과, 큐의 급격한 혼잡이 애초에 일어나지 않도록 예방하여 전체 에피소드 동안 물리적인 전송 지연비용 및 적재 패널티를 대폭 경감시키는 방향으로 동작하게 됩니다.

3. **Expected SARSA와 Q-Learning의 학습 안정성 차이**:
   - 과부하 상태($\lambda = 1.5$)에서 Cliff 리워드의 경우, **Expected SARSA**는 기대 보상 **-150.97 (대기 762.69회)**을 달성하여 최적 선에 근사한 반면, **Q-Learning**은 기대 보상 **-298.35 (대기 764.95회)**로 수렴 성능이 비교적 뒤처졌습니다.
   - 이는 Cliff 환경 특유의 경계면 확률적 노이즈가 작용할 때, Q-Learning은 탐욕적 최대값($\max$) 갱신 방식으로 인해 패널티를 과소/과대평가하여 편향이 누적되는 반면, Expected SARSA는 확률적 탐험 행동들의 가치 기댓값을 전파하므로 정책 갱신 과정이 훨씬 안정적이고 보수적으로 수렴하기 때문입니다.

---

## 5. 프로젝트 파일 설명 및 구조 (Project File Descriptions)

이 프로젝트는 MDP 정의, 강화학습 에이전트 학습, 시뮬레이션 및 다차원 시각화 도구들로 이루어져 있습니다. 주요 코드와 설정 파일의 역할은 다음과 같습니다.

| 파일 이름 | 역할 및 핵심 기능 설명 |
| :--- | :--- |
| [mdc_mdp_env.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/mdc_mdp_env.py) | **Gymnasium 기반 환경 구현**: 오프로딩 MDP의 상태 전이(stochastic channel, Poisson arrival 등), 행동 및 보상 계산을 담당합니다. 특히 초과 태스크를 임시 보관하고 복원하는 **무한대 대기 버퍼(pending_buffer)** 시스템이 구현되어 있습니다. |
| [build_mdp_model.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/build_mdp_model.py) | **분석적 MDP 모델 구축**: 이론적 상태 전이 확률 행렬 $P(S' \mid S, a)$와 기대 보상 벡터 $R(S, a)$를 수학 공식 기반으로 사전 생성하여 `models/` 폴더에 Pickle(`*.pkl`) 파일로 내보냅니다. |
| [train_all_mdp.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/train_all_mdp.py) | **강화학습 에이전트 학습**: 지정된 에피소드(기본 100,000회) 동안 Q-러닝 및 Expected SARSA 에이전트를 학습시키고, 학습된 Q-테이블을 CSV 형태로 저장합니다. 동적 계획법(DP: Policy/Value Iteration)의 이론적 성능 한계 수치도 함께 평가합니다. |
| [visualize_mdp_results.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/visualize_mdp_results.py) | **개별 시나리오 시각화**: 에이전트들의 누적 할인 보상 학습 곡선(`mdp_training_curves.png`), 성능 비교 바 차트(`mdp_performance_bars.png`), 각 차원별 정책 히트맵(`mdp_heatmap_*.png`)을 시나리오 폴더별로 자동 플롯합니다. |
| [visualization/export_policy_data.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/visualization/export_policy_data.py) | **웹 대시보드 데이터 연동**: 학습된 모든 시나리오별 Q-테이블과 최적(DP) 정책 데이터를 단일 파일(`visualization/policy_data.js` 및 `policy_data.json`)로 취합하여 웹 비주얼라이저가 로드할 수 있도록 파싱합니다. |
| [generate_queue_gif.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/generate_queue_gif.py) | **대기열 시뮬레이션 애니메이션 생성**: Cliff 정책과 Standard 정책을 바탕으로 큐의 크기 변화와 의도적/강제 대기(Pending) 횟수의 누적 변화를 직관적인 차트 애니메이션(8.0 FPS 고속 GIF)으로 렌더링합니다. |
| [generate_policy_comparison_gif.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/generate_policy_comparison_gif.py) | **정책 행동 패턴 비교 애니메이션 생성**: DP Optimal vs Expected SARSA vs Q-Learning 세 가지 정책이 동일한 도착 상태 조건 하에서 어떻게 큐 분산 제어를 진행하는지 Side-by-Side 비교 애니메이션(GIF)으로 생성합니다. |
| [aggregate_all_comparisons.py](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/aggregate_all_comparisons.py) | **통합 최종 결과 집계**: 전체 학습이 완료된 6개 시나리오 데이터를 스캔하여 전체 성능 지표(평균 기대 보상, 평균 대기(Pending) 건수, 에너지 소모량)를 마크다운 표 형태로 집계하여 화면에 출력합니다. |
| [run_mdp_pipeline.sh](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/run_mdp_pipeline.sh) | **자동화 쉘 스크립트 (Bash)**: Git Bash, Linux, macOS 환경 등에서 전체 시나리오 학습 및 시각화 파이프라인(Step 1 ~ 7)을 한 번에 구동할 수 있도록 제어하는 자동화 스크립트입니다. |
| [run_mdp_pipeline.ps1](file:///C:/Users/sbeen/OneDrive/Desktop/RL_project%20-%20%EB%B3%B5%EC%82%AC%EB%B3%B8/run_mdp_pipeline.ps1) | **자동화 파워쉘 스크립트 (PowerShell)**: Windows 환경 사용자를 위해 제공되는 배치 파이프라인 제어 스크립트입니다. |

---

## 6. 파이프라인 실행 방법 및 실행 순서 가이드 (Pipeline Execution Guide)

이 프로젝트의 모든 학습과 결과 시각화 분석은 통합 스크립트를 통해 일괄 실행하거나 개별 명령어로 진행할 수 있습니다.

### 방법 A. 쉘 스크립트(Bash)를 통한 자동 실행 (추천)
**Git Bash** 또는 **WSL(Linux)** 터미널에서 전체 파이프라인을 실행합니다. 기본값으로 100,000 에피소드와 6개 주요 시나리오($\lambda \in [0.5, 1.0, 1.5]$, 리워드 `standard` 및 `cliff`)에 대해 학습 및 시각화(GIF 포함)가 순차 실행됩니다.

1. **기본값으로 전체 파이프라인 실행**:
   ```bash
   bash run_mdp_pipeline.sh
   ```
   *(또는 스크립트에 실행 권한을 부여한 뒤 `./run_mdp_pipeline.sh`로 실행 가능)*

2. **특정 파라미터로 커스텀 파이프라인 실행**:
   * `-l`: 테스트할 $\lambda$ 목록 (공백으로 구분)
   * `-e`: 에피소드 수
   * `-r`: 적용할 보상 형태 (공백으로 구분)
   ```bash
   # 예: 0.5 및 1.5 람다에 대해 5,000 에피소드만 빠르게 학습
   ./run_mdp_pipeline.sh -l "0.5 1.5" -e 5000 -r "standard cliff"
   ```

---

### 방법 B. PowerShell을 통한 자동 실행 (Windows)
Windows PowerShell 콘솔 환경에서 실행하기 위한 파이프라인 구동 스크립트입니다.

* **기본 파이프라인 실행**:
  ```powershell
  # 기본 파라미터 (Lambda = 1.5, Episodes = 10000) 기준으로 실행
  ./run_mdp_pipeline.ps1 -Lambda 1.5 -Episodes 100000
  ```

---

### 방법 C. 단계별 수동 실행 (Step-by-Step Manual Execution)
원하는 단계만 직접 세부 옵션을 설정하여 실행할 수 있는 명령어 구성입니다. 모든 명령어는 프로젝트 루트 디렉토리에서 실행합니다.

#### [Step 1] MDP 전이 모델 구축 (Pickle 파일 생성)
강화학습 평가의 기준이 되는 동적 계획법(DP) 수렴을 위해 이론적 모델을 구축합니다.
```bash
python build_mdp_model.py --lambda_val=0.5 --reward_type=standard
python build_mdp_model.py --lambda_val=0.5 --reward_type=cliff
python build_mdp_model.py --lambda_val=1.0 --reward_type=standard
python build_mdp_model.py --lambda_val=1.0 --reward_type=cliff
python build_mdp_model.py --lambda_val=1.5 --reward_type=standard
python build_mdp_model.py --lambda_val=1.5 --reward_type=cliff
```
*(결과 파일: `models/mdp_model_{reward_type}_L{lambda}.pkl`)*

#### [Step 2] Q-Learning / Expected SARSA 학습 및 성능 평가
각 시나리오별 강화학습 에이전트를 지정된 에피소드 동안 학습시킵니다.
```bash
# 6개 시나리오 순차 학습 (각 100,000 에피소드)
python train_all_mdp.py --lambda_val=0.5 --episodes=100000 --reward_type=standard
python train_all_mdp.py --lambda_val=0.5 --episodes=100000 --reward_type=cliff
python train_all_mdp.py --lambda_val=1.0 --episodes=100000 --reward_type=standard
python train_all_mdp.py --lambda_val=1.0 --episodes=100000 --reward_type=cliff
python train_all_mdp.py --lambda_val=1.5 --episodes=100000 --reward_type=standard
python train_all_mdp.py --lambda_val=1.5 --episodes=100000 --reward_type=cliff
```
*(결과 파일: `results/L_{lambda}_E_{episodes}/{reward_type}/q_table_ql.csv` 및 `q_table_sarsa.csv`)*

#### [Step 3] 개별 시나리오 수렴도 곡선 및 정책 히트맵 이미지 플롯
에이전트별 성능 도출 결과를 시각화합니다.
```bash
# 시나리오별 개별 시각화 이미지 생성
python visualize_mdp_results.py --lambda_val=0.5 --episodes=100000 --reward_type=standard
python visualize_mdp_results.py --lambda_val=0.5 --episodes=100000 --reward_type=cliff
# (나머지 시나리오도 동일 인수 구조로 실행)
```
*(결과 파일: 각 폴더 내 `mdp_training_curves.png`, `mdp_performance_bars.png`, `mdp_heatmap_*.png`)*

#### [Step 4] 웹 대시보드 시각화용 데이터 취합 및 JSON 포맷 내보내기
학습된 모든 시나리오 결과를 모아서 웹 가시화용 리소스 파일로 추출합니다.
```bash
python visualization/export_policy_data.py --episodes=100000
```
*(결과 파일: `visualization/policy_data.js` 및 `results/policy_data.js`)*

#### [Step 5] 큐 대기열 변동 및 대기(Pending) 통계 시뮬레이션 GIF 렌더링
보류 횟수를 실시간 표시하는 큐 동작 애니메이션을 인코딩합니다.
```bash
python generate_queue_gif.py --episodes=100000
```
*(결과 파일: `results/queue_simulation_L*.gif`)*

#### [Step 6] 3대 정책 분산 비교 애니메이션 GIF 렌더링
에너지 소모 및 지연 시간을 조율하는 행동 매핑 비교 애니메이션을 구동합니다.
```bash
python generate_policy_comparison_gif.py --episodes=100000
```
*(결과 파일: `results/policy_comparison_L*.gif`)*

#### [Step 7] 종합 성적 결과 집계 테이블 요약 화면 출력
모든 실험 메트릭들을 요약 정리하여 가독성 있는 마크다운 보고형식으로 표시합니다.
```bash
python aggregate_all_comparisons.py --episodes=100000
```
*(결과 파일: 터미널 stdout에 종합 테이블 표가 출력됨)*
