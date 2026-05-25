@echo off
echo === Starting Short Verification Run (Lambda=0.5, Episodes=10, Reward=standard) ===

echo [Step 1] Building MDP Model...
python build_mdp_model.py --lambda_val=0.5 --reward_type=standard
if %errorlevel% neq 0 (echo Step 1 failed & exit /b %errorlevel%)

echo [Step 2] Training Agents (10 episodes)...
python train_all_mdp.py --lambda_val=0.5 --episodes=10 --reward_type=standard
if %errorlevel% neq 0 (echo Step 2 failed & exit /b %errorlevel%)

echo [Step 3] Visualizing Results...
python visualize_mdp_results.py --lambda_val=0.5 --episodes=10 --reward_type=standard
if %errorlevel% neq 0 (echo Step 3 failed & exit /b %errorlevel%)

echo [Step 4] Exporting Policy Data...
python visualization/export_policy_data.py --episodes=10
if %errorlevel% neq 0 (echo Step 4 failed & exit /b %errorlevel%)

echo [Step 5] Generating Queue Simulation GIFs...
python generate_queue_gif.py --episodes=10
if %errorlevel% neq 0 (echo Step 5 failed & exit /b %errorlevel%)

echo [Step 6] Generating Policy Comparison GIFs...
python generate_policy_comparison_gif.py --episodes=10
if %errorlevel% neq 0 (echo Step 6 failed & exit /b %errorlevel%)

echo [Step 7] Aggregating All Results...
python aggregate_all_comparisons.py --episodes=10
if %errorlevel% neq 0 (echo Step 7 failed & exit /b %errorlevel%)

echo === Verification Run Complete ===
