$ErrorActionPreference = "Stop"

# Submission lock profile (does not modify local .env file).
$env:SUBMISSION_MODE = "1"
$env:PROMPT_VERSION = "v2"
$env:FALLBACK_MODEL_NAME = ""
$env:MODEL_FALLBACKS = ""
$env:USE_CLIENT_RECONNECT_WRAPPER = "1"
$env:INTERACTIVE_MODE = "0"
$env:ENSEMBLE_VOTES = "1"
$env:WARM_START_STEPS = "0"
$env:EMIT_PROJECTED_LOGS = "0"
$env:EMIT_TOKEN_LOGS = "0"
$env:NOOP_STREAK_THRESHOLD = "3"
$env:PROMPT_MAX_CHARS = "12000"

Write-Host "[1/2] Running inference.py with submission profile"
python inference.py

Write-Host "[2/2] Running pre-submit checks"
python scripts/pre_submit_check.py

Write-Host "[OK] Submission profile run completed."
