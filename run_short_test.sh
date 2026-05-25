#!/bin/bash
echo "=== Starting Short Verification Run (Lambda=0.5, Episodes=10, Reward=standard) ==="

# Step 1: Build model
echo "[Step 1] Building MDP Model..."
python build_mdp_model.py --lambda_val=0.5 --reward_type=standard
if [ $? -ne 0 ]; then echo "Step 1 failed"; exit 1; fi

# Step 2: Train and evaluate
echo "[Step 2] Training Agents (10 episodes)..."
python train_all_mdp.py --lambda_val=0.5 --episodes=10 --reward_type=standard
if [ $? -ne 0 ]; then echo "Step 2 failed"; exit 1; fi

# Step 3: Visualize
echo "[Step 3] Visualizing Results..."
python visualize_mdp_results.py --lambda_val=0.5 --episodes=10 --reward_type=standard
if [ $? -ne 0 ]; then echo "Step 3 failed"; exit 1; fi

# Step 4: Export Policy Data
echo "[Step 4] Exporting Policy Data..."
python visualization/export_policy_data.py --episodes=10
if [ $? -ne 0 ]; then echo "Step 4 failed"; exit 1; fi

# Step 5: Generate Queue Simulation GIF
echo "[Step 5] Generating Queue Simulation GIFs..."
python generate_queue_gif.py --episodes=10
if [ $? -ne 0 ]; then echo "Step 5 failed"; exit 1; fi

# Step 6: Generate Policy Comparison GIF
echo "[Step 6] Generating Policy Comparison GIFs..."
python generate_policy_comparison_gif.py --episodes=10
if [ $? -ne 0 ]; then echo "Step 6 failed"; exit 1; fi

# Step 7: Aggregate all results
echo "[Step 7] Aggregating All Results..."
python aggregate_all_comparisons.py --episodes=10
if [ $? -ne 0 ]; then echo "Step 7 failed"; exit 1; fi

echo "=== Verification Run Complete ==="
