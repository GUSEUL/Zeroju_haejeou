#!/bin/bash

# 기본값 설정
LAMBDA=1.5
EPISODES=5000
REWARD_TYPES=("standard" "sparse" "cliff")

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
    l) LAMBDA=$OPTARG ;;
    e) EPISODES=$OPTARG ;;
    r) REWARD_TYPES=($OPTARG) ;; # 특정 reward_type만 지정 가능
    \?) echo "사용법: $0 [-l lambda] [-e episodes] [-r reward_type]" >&2; exit 1 ;;
  esac
done

for RT in "${REWARD_TYPES[@]}"; do
    echo ""
    echo "=========================================================="
    echo " MDC MDP Pipeline: [Lambda: $LAMBDA, Reward: $RT]"
    echo "=========================================================="

    # Step 1: MDP 모델 빌드
    echo "[Step 1] Building MDP Model..."
    $PYTHON_CMD build_mdp_model.py --lambda_val=$LAMBDA --reward_type=$RT
    if [ $? -ne 0 ]; then echo "Step 1 실패 ($RT)"; continue; fi

    # Step 2: 알고리즘 학습 및 평가
    echo "[Step 2] Training and Evaluating Agents..."
    $PYTHON_CMD train_all_mdp.py --lambda_val=$LAMBDA --episodes=$EPISODES --reward_type=$RT
    if [ $? -ne 0 ]; then echo "Step 2 실패 ($RT)"; continue; fi

    # Step 3: 결과 시각화
    echo "[Step 3] Visualizing Results..."
    $PYTHON_CMD visualize_mdp_results.py --lambda_val=$LAMBDA --reward_type=$RT
    if [ $? -ne 0 ]; then echo "Step 3 실패 ($RT)"; continue; fi

done

echo ""
echo "=========================================================="
echo " 모든 리워드 타입에 대한 실험이 완료되었습니다."
echo "=========================================================="
