# Run all MDC simulation and visualization scripts in sequence

Write-Host "[1/4] Running compare_baselines.py..."
python MDC/compare_baselines.py
if ($LASTEXITCODE -ne 0) { Write-Error "compare_baselines.py failed"; exit $LASTEXITCODE }

Write-Host "`n[2/4] Running compare_unstable_network.py..."
python MDC/compare_unstable_network.py
if ($LASTEXITCODE -ne 0) { Write-Error "compare_unstable_network.py failed"; exit $LASTEXITCODE }

Write-Host "`n[3/4] Running visualize_network.py..."
Write-Host "NOTE: This will open a Pygame window. Please close it manually to proceed to the next script."
python MDC/visualize_network.py
if ($LASTEXITCODE -ne 0) { Write-Error "visualize_network.py failed"; exit $LASTEXITCODE }

Write-Host "`n[4/4] Running visualize_results.py..."
python MDC/visualize_results.py
if ($LASTEXITCODE -ne 0) { Write-Error "visualize_results.py failed"; exit $LASTEXITCODE }

Write-Host "`n[Success] All MDC scripts have been executed."
