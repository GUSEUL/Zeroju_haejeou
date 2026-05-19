
# MDC RL Offloading Project

이 프로젝트는 모바일 분산 컴퓨팅(Mobile Distributed Computing, MDC) 환경에서의 태스크 오프로딩 전략을 강화학습(RL)으로 최적화하는 프레임워크를 제공합니다. 커스텀 Gymnasium 환경, 다양한 베이스라인 에이전트, 그리고 이론적 최적값 분석을 위한 MDP 도구들을 포함하고 있습니다.

## 1. 기술 명세 (Technical Specifications)

### 상태 공간 (State Space, $\mathcal{S}$)
관측 가능한 상태는 다음 7개의 변수로 구성된 이산 벡터입니다 (총 4,356개 상태):
$S = [Task, Channel, CPU\_BW, Local\_Q, Best\_Neighbor\_Q]$
- **Task**: 태스크 유형 (0: URLLC, 1: eMBB)
- **Channel**: 채널 품질 (0: Poor, 1: Normal, 2: Good)
- **CPU_BW**: 통합 리소스 상태 (0-5: CPU 부하 3단계 $\times$ 대역폭 2단계)
- **Local_Q**: 로컬 큐에 쌓인 태스크 수 (0-10)
- **Best_Neighbor_Q**: 가장 한가한 주변 노드의 큐 상태 (0-10)

### 행동 공간 (Action Space, $\mathcal{A}$)
에이전트는 매 스텝 4가지 행동 중 하나를 선택합니다:
- $a = 0$: **로컬 처리 (Local Processing)**
- $a = 1$: **주변 노드 1로 오프로딩 (Offload to N1)**
- $a = 2$: **주변 노드 2로 오프로딩 (Offload to N2)**
- $a = 3$: **의도적 드롭 (Intentional Drop)**

### 보상 함수 (Reward Function, $R$)
지연 시간, 큐 혼잡도, 에너지 효율을 종합적으로 고려합니다:
$R = r_{\text{delay}} + p_{q} + r_{\text{energy}} + r_{\text{drop}}$

---

## 2. 주요 폴더 및 파일 구성

### ?? Core Project (MDC 폴더)
실용적인 강화학습 학습 및 시뮬레이션을 위한 코드입니다.
- **`mdc_gym_env.py`**: 기본적인 Gymnasium 환경 정의.
- **`compare_baselines.py`**: SARSA, Q-Learning 학습 및 베이스라인 비교.
- **`visualize_results.py`**: 학습 결과 바 차트 및 수렴 곡선 생성.
- **`visualize_network.py`**: Pygame 기반의 실시간 네트워크 시뮬레이션 시각화.

### ?? Analytical MDP Suite (MDP 폴더)
이론적 최적해(DP) 도출 및 정밀 비교 분석을 위한 스크립트 세트입니다.

- **`mdc_mdp_env.py`**: 완벽한 마르코프 속성을 위해 정수형 큐를 사용하는 정밀 MDP 환경.
- **`build_mdp_model.py`**: DP 계산에 필수적인 **전이 확률 행렬($P$)** 및 **기대 보상 행렬($R$)** 생성.
- **`train_all_mdp.py`**: 7종 알고리즘(Value/Policy Iteration, MC, SARSA, QL, Baselines) 통합 실행 및 **학습 시간** 측정.
- **`visualize_mdp_results.py`**: 분석 결과 CSV를 바탕으로 **성능 비교 바 차트** 및 **최적 정책 히트맵** 생성.
- **`run_mdp_pipeline.ps1`**: 전체 분석 과정(빌드 -> 학습 -> 시각화) 자동화 스크립트.

---

## 3. 핵심 분석 지표 (Performance Metrics)

분석 결과(`mdp_final_results.csv`)에서 다음 지표들을 통해 알고리즘을 평가합니다:
- **reward**: 에피소드당 평균 총 보상 (시스템 효율성)
- **drops**: 에피소드당 평균 태스크 드롭 수 (신뢰성)
- **energy**: 에피소드당 평균 에너지 소모량 (지속 가능성)
- **time**: 최적 정책 도출까지 걸린 실제 연산 시간 (계산 복잡도)

---

## 4. 시작하기 (Getting Started)

### 환경 설정
```bash
pip install gymnasium pygame numpy pandas matplotlib seaborn
```

### MDP 분석 파이프라인 실행
1. `cd MDP`
2. `./run_mdp_pipeline.ps1 -Lambda 1.5 -Episodes 10000`

