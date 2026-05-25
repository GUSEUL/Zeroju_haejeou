#!/bin/bash

# 기본값 설정
LAMBDAS=(0.5 1.0 1.5)
EPISODES=100000
REWARD_TYPES=("standard" "cliff")

# Python 명령어 확인
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "에러: Python이 설치되어 있지 않습니다."
        exit 1
    fi
fi

# 인자 처리
while getopts "l:e:r:" opt; do
  case $opt in
    l) LAMBDAS=($OPTARG) ;;
    e) EPISODES=$OPTARG ;;
    r) REWARD_TYPES=($OPTARG) ;;
    \?) echo "사용법: $0 [-l 'lambda_list'] [-e episodes] [-r 'reward_types']" >&2; exit 1 ;;
  esac
done

for LAMBDA in "${LAMBDAS[@]}"; do
    for RT in "${REWARD_TYPES[@]}"; do
        echo ""
        echo "=========================================================="
        echo " MDC MDP Pipeline: [Lambda: $LAMBDA, Reward: $RT]"
        echo "=========================================================="

        # Step 1: MDP 모델 빌드 (정밀도 향상: samples=2000)
        echo "[Step 1] Building MDP Model..."
        $PYTHON_CMD build_mdp_model.py --lambda_val=$LAMBDA --reward_type=$RT
        if [ $? -ne 0 ]; then echo "Step 1 실패 ($RT, L=$LAMBDA)"; continue; fi

        # Step 2: 알고리즘 학습 및 평가 (정확도 향상: eval=500)
        echo "[Step 2] Training and Evaluating Agents..."
        $PYTHON_CMD train_all_mdp.py --lambda_val=$LAMBDA --episodes=$EPISODES --reward_type=$RT
        if [ $? -ne 0 ]; then echo "Step 2 실패 ($RT, L=$LAMBDA)"; continue; fi

        # Step 3: 결과 시각화
        echo "[Step 3] Visualizing Results..."
        $PYTHON_CMD visualize_mdp_results.py --lambda_val=$LAMBDA --episodes=$EPISODES --reward_type=$RT
        if [ $? -ne 0 ]; then echo "Step 3 실패 ($RT, L=$LAMBDA)"; continue; fi
    done
done

# Step 4: 웹 대시보드 시각화용 데이터 파일 내보내기
echo ""
echo "[Step 4] Exporting Policy Data..."
$PYTHON_CMD visualization/export_policy_data.py --episodes=$EPISODES

# Step 5: 고속 큐 시뮬레이션 애니메이션 (GIF) 생성
echo "[Step 5] Generating Queue Simulation GIFs..."
$PYTHON_CMD generate_queue_gif.py --episodes=$EPISODES

# Step 6: 3대 정책 side-by-side 비교 애니메이션 (GIF) 생성
echo "[Step 6] Generating Policy Comparison GIFs..."
$PYTHON_CMD generate_policy_comparison_gif.py --episodes=$EPISODES

# Step 7: 통합 최종 결과 요약 테이블 생성
echo "[Step 7] Aggregating All Results..."
$PYTHON_CMD aggregate_all_comparisons.py --episodes=$EPISODES

echo ""
echo "=========================================================="
echo " 모든 시나리오 및 후속 시뮬레이션 시각화가 완료되었습니다."
echo " 결과 폴더(results/)를 확인해주세요."
echo "=========================================================="
