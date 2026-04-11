$ErrorActionPreference = "Stop"

Write-Host "[1/2] Running inference.py"
python inference.py

Write-Host "[2/2] Running pre-submit checks"
python scripts/pre_submit_check.py

Write-Host "[OK] Full pre-submit flow completed."
