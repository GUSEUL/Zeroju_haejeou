
# MDP Analysis Pipeline Automation Script
param (
    [double]$Lambda = 1.5,
    [int]$Episodes = 10000
)

Write-Host \"`n>>> Step 1: Building Analytical MDP Model (Lambda = $Lambda) <<<\" -ForegroundColor Cyan
python build_mdp_model.py --lambda_val=$Lambda

Write-Host \"`n>>> Step 2: Training & Evaluating All Algorithms (Episodes = $Episodes) <<<\" -ForegroundColor Cyan
python train_all_mdp.py --lambda_val=$Lambda --episodes=$Episodes

Write-Host \"`n>>> Step 3: Visualizing Results <<<\" -ForegroundColor Cyan
python visualize_mdp_results.py --lambda_val=$Lambda

Write-Host \"`n[Success] Pipeline execution complete. Check PNG and CSV files for results.\" -ForegroundColor Green

