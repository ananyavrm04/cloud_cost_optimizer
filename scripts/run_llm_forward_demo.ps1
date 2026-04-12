$ErrorActionPreference = "Stop"

# Optional demo profile: encourages more LLM-led steps for analysis/presentation.
# Do NOT use this as your locked submission profile.
$env:SUBMISSION_MODE = "1"
$env:PROMPT_VERSION = "v2"
$env:FALLBACK_MODEL_NAME = ""
$env:MODEL_FALLBACKS = ""
$env:USE_CLIENT_RECONNECT_WRAPPER = "1"
$env:INTERACTIVE_MODE = "0"
$env:ENSEMBLE_VOTES = "1"
$env:WARM_START_STEPS = "0"
$env:NOOP_STREAK_THRESHOLD = "999"
$env:NEGATIVE_STREAK_THRESHOLD = "999"
$env:FORCE_LLM_EVERY_STEP = "1"
$env:EMIT_PROJECTED_LOGS = "1"
$env:EMIT_TOKEN_LOGS = "1"
$env:PROMPT_MAX_CHARS = "12000"

Write-Host "[1/2] Running inference.py with LLM-forward demo profile"
python inference.py

Write-Host "[2/2] Running pre-submit checks"
python scripts/pre_submit_check.py

Write-Host "[OK] LLM-forward demo run completed."
