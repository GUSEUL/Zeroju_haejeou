# MDC MDP Task Offloading Project - Code Instruction

이 파일은 `MDP` 디렉토리에 포함된 모든 파이썬 파일의 상세 역할과 생성되는 결과물(Output)에 대해 설명합니다.

## 1. 주요 파이썬 스크립트 상세 설명

### 🟢 `mdc_mdp_env.py`
- **역할**: MDP(Markov Decision Process) 모델링을 위해 커스텀 제작된 Gymnasium 환경입니다.
- **주요 구성**:
    - **상태 공간(Observation Space)**: `[Task, Comm_state, LocalQ, N1_Q, N2_Q]` (총 3,630개 상태)
        - **Task (2)**: 0(URLLC), 1(eMBB)
        - **Comm_state (3)**: 통신 및 리소스 종합 상태 (0:불량, 1:보통, 2:좋음)
        - **LocalQ (5)**: 로컬 큐 용량 (0~4)
        - **N1_Q, N2_Q (11)**: 이웃 노드 1, 2의 각각의 큐 용량 (0~10)
    - **액션 공간(Action Space)**: `0: Local`, `1: Neighbor 1`, `2: Neighbor 2`, `3: Intentional Drop`
    - **보상 함수 (Reward Function)**: $R = r_{delay} + p_{q} + r_{energy} + r_{drop}$
        - $p_{q} = -(\exp(2.0 \cdot Queue/5) - 1)$ (로컬 큐 5개 용량 기준 페널티)

### 🔵 `build_mdp_model.py`
- **역할**: 환경의 전이 확률 행렬($P$)과 기대 보상($R$)을 수치적으로 계산하여 MDP 모델을 생성합니다.
- **출력 (Output)**:
    - `mdp_model_L{lambda}.pkl`: 3,630개 상태에 대한 전이 및 보상 데이터가 포함된 Pickle 파일.

### 🔴 `train_all_mdp.py`
- **역할**: MDP 모델을 바탕으로 최적 정책(DP)과 강화학습(RL) 알고리즘의 성능을 비교 평가합니다.

### 📊 `visualize_mdp_results.py`
- **역할**: 학습 및 평가 결과를 시각화하여 분석 보고서용 이미지를 생성합니다.

---

## 2. 자동화 및 실행 도구

### 🛠️ `run_mdp_pipeline.ps1`
- **역할**: 모델 빌드부터 시각화까지 전 과정을 자동화하는 PowerShell 스크립트입니다.
- **사용법**:
  ```powershell
  ./run_mdp_pipeline.ps1 -Lambda 1.5 -Episodes 10000
  ```
