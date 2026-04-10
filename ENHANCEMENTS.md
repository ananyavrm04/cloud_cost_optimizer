# Cloud Cost Optimizer - Enhancement Backlog (100 Items)

This document tracks a complete list of proposed improvements for performance, reliability, validator compatibility, and production readiness.

## 1. Deterministic LLM Output Parsing
- Enforce strict JSON extraction/parsing for LLM responses.
- Add one corrective reprompt when JSON is malformed.
- Reduce random parse failures and invalid action payloads.

## 2. Budget-Aware Action Policy
- Add limits for API calls, step usage, and token budget per task.
- Stop early when expected marginal gain is very low.
- Improve completion reliability under credit constraints.

## 3. Two-Stage Decisioning
- Stage 1: rank candidate resources.
- Stage 2: choose action only among top-k candidates.
- Improve quality/consistency on medium and hard tasks.

## 4. Critical Dependency Graph Safety
- Build a dependency DAG (or equivalent graph) each step.
- Prevent risky terminate/resize on upstream critical nodes.
- Protect SLA and reduce violation-related score penalties.

## 5. Action Confidence Gating
- Require model to output a confidence score.
- If confidence is below threshold, use safe heuristic action.
- Improve robustness against uncertain LLM outputs.

## 6. Task-Specific Prompt Templates
- Use dedicated prompts for `easy`, `medium`, and `hard`.
- Encode task-specific constraints and common failure patterns.
- Improve action quality and convergence speed.

## 7. Structured Diagnostics (Non-Breaking)
- Keep required `[START]/[STEP]/[END]` format unchanged.
- Add optional compact debug lines (for local runs only).
- Speed up debugging while preserving validator compatibility.

## 8. Local Validator Script
- Create a pre-submit validator script for:
  - required env vars
  - stdout format
  - score range checks
  - proxy call path verification
- Catch failures before submission.

## 9. Regression Test Suite for Inference
- Add tests for parsing, normalization, scoring, and logging.
- Add regression fixtures for known failure modes.
- Prevent accidental breakage during iteration.

## 10. Submission Mode Flag
- Add `SUBMISSION_MODE=1` to enforce strict validator-safe behavior.
- Lock required env usage, output structure, and score formatting.
- Avoid local-debug settings leaking into submission runs.

## 11. LLM Network Resilience
- Add timeout and retry handling around `call_llm()`.
- Catch API/network exceptions and fallback safely.
- Prevent episode hangs and improve reliability.

## 12. Step History in Prompt
- Include a rolling window of recent actions/rewards in prompt.
- Give agent short-term memory of prior attempts.
- Reduce repeated failed/duplicate actions.

## 13. Dependency Check for Resize
- Apply dependent-resource safety checks to resize logic as well.
- Align with terminate safety guard behavior.
- Reduce SLA risk from downsizing upstream dependencies.

## 14. Semantic Validation in normalize_action
- Validate action semantics, not just `resource_id` existence.
- Reject impossible/no-op actions before `env.step()`.
- Reduce wasted steps and penalty actions.

## 15. Max Steps Scaling for Hard Task
- Increase step budget policy (or make configurable per task).
- Account for multi-action optimization paths on same resource.
- Reduce premature episode termination on hard scenarios.

## 16. Already-Optimized Resource Tracking
- Track resources already acted upon (terminate/resize/switch).
- Avoid repeating penalized actions on same IDs.
- Improve step efficiency and score stability.

## 17. Parallel Task Execution
- Run `easy`, `medium`, `hard` concurrently where safe.
- Use `ThreadPoolExecutor` or equivalent orchestration.
- Reduce wall-clock run time (with careful stdout ordering controls).

## 18. Score Calibration Sanity Band
- Revisit low-end score handling when SLA is violated.
- Optionally define a clearer penalty band while keeping `(0,1)` rule.
- Improve interpretability of failed-SLA outcomes.

## 19. Spot Pricing in Heuristic
- Add `spot` pricing switch for non-critical eligible resources.
- Use risk-aware rule (critical/dependency constraints).
- Capture additional cost savings beyond reserved-only policy.

## 20. Observation Schema Version Field
- Add `schema_version` to observation model (e.g., `"1.0"`).
- Detect client/server schema drift early.
- Improve long-term compatibility and debugging.

## 21. Early done=True on Full Optimization
- End episode early when optimal savings target is reached.
- Avoid unnecessary extra steps after full optimization.
- Improve efficiency and reduce noisy actions.

## 22. Task Metadata in Observation/Prompt
- Surface progress context such as:
  - `total_savings`
  - `optimal_savings`
  - `savings_progress`
- Help model decide when to stop vs continue optimizing.

## 23. Multi-Turn LLM Context Window
- Maintain a sliding context window of recent actions and feedback.
- Help the LLM remember what already worked or failed.
- Reduce repetitive low-value decisions.

## 24. Action Ordering Optimizer
- Replace rigid priority with ranked candidate actions.
- Score candidates by expected savings versus risk.
- Improve step efficiency on medium/hard tasks.

## 25. Resource Type-Specific Strategy
- Use different rules for compute, storage, and database.
- Apply stricter safety on high-risk resource types.
- Improve SLA safety without losing savings.

## 26. Risk Score Per Action
- Compute risk using criticality, dependents, type, and uptime margin.
- Include this risk in prompt/action selection.
- Make tradeoffs explicit and safer.

## 27. Episode Checkpointing and Replay
- Store per-step observation, action, reward, and done state.
- Save traces as JSON for offline debugging.
- Build reusable datasets from real runs.

## 28. Composite Action Support
- Allow applying multiple compatible changes in one logical action.
- Example: resize plus pricing switch on same resource.
- Improve step efficiency where rules permit.

## 29. Cost-of-Change Modeling
- Add one-time transition costs for resize/pricing migration.
- Model temporary risk/downtime penalties.
- Make simulation closer to real cloud operations.

## 30. Uptime Margin Awareness in Prompt
- Explicitly include `uptime_margin = current_uptime - sla_target`.
- Help LLM decide how aggressive it can be.
- Reduce accidental SLA breaches.

## 31. Progressive Difficulty Mode
- Chain tasks with auto-advance based on score threshold.
- Test strategy adaptation across changing difficulty.
- Enable curriculum-style run mode.

## 32. Environment Stochasticity
- Add optional random fluctuation to utilization metrics.
- Prevent overfitting to static observations.
- Increase robustness to noisy real-world patterns.

## 33. Action Undo and Rollback
- Support undo for previous risky actions.
- Allow recovery path after mistakes.
- Improve resilience during exploratory optimization.

## 34. Token Usage Tracking
- Record prompt/completion/total tokens per LLM call.
- Emit token statistics in logs.
- Support inference cost optimization.

## 35. Dependency Graph Visualization
- Build dependency DAG summary at episode start.
- Log chain structures for debugging.
- Improve observability in hard tasks.

## 36. Savings Velocity Metric
- Track savings gain per step.
- Detect flat optimization phases.
- Auto-stop when additional steps are not productive.

## 37. Custom Task Builder CLI
- Provide CLI to generate task JSON by parameters.
- Control resource count, critical ratio, and dependency depth.
- Enable custom benchmarks beyond bundled tasks.

## 38. LLM Response Caching
- Cache prompt hash to action response.
- Reuse responses on identical observations.
- Reduce redundant calls and token usage.

## 39. Multi-Agent Comparison Mode
- Run multiple model/prompt configs on same tasks.
- Output side-by-side performance comparison.
- Accelerate strategy benchmarking.

## 40. Graceful Degradation on API Failure
- Detect persistent API failure and switch mode safely.
- Continue episode with fallback policy when allowed.
- Avoid hard crashes and partial logs.

## 41. Few-Shot Examples in Prompt
- Add compact worked examples in system prompt.
- Demonstrate correct action selection patterns.
- Improve first-step decision quality.

## 42. Observation Diffing
- Compute and include what changed since previous step.
- Highlight savings and state deltas explicitly.
- Reduce cognitive load on large observations.

## 43. Chain-of-Thought Extraction
- Request concise rationale field separate from action JSON.
- Log rationale for debugging and model analysis.
- Keep final action output machine-parseable.

## 44. Temperature Annealing
- Start deterministic, increase exploration only when stuck.
- Use controlled schedule tied to progress.
- Balance stability and discovery.

## 45. Action Deduplication Guard
- Block immediate repeats of recently failed identical actions.
- Prevent repeated penalty loops.
- Preserve step budget.

## 46. Prompt Compression for Hard Task
- Summarize repetitive resources into compact groups.
- Keep important IDs and aggregate stats.
- Improve relevance under context limits.

## 47. Step Budget Awareness
- Include `steps_remaining` in prompt.
- Prioritize high-value actions near episode end.
- Reduce wasted low-impact moves.

## 48. Reward Prediction Before Acting
- Estimate expected gain locally before submitting action.
- Skip clearly negative/no-op actions.
- Improve action quality and efficiency.

## 49. Cost Breakdown by Category
- Add compute/storage/database monthly subtotal in prompt.
- Focus model on largest cost buckets first.
- Improve prioritization.

## 50. Resource Priority Ranking
- Present resources sorted by potential impact.
- Put high-cost/high-opportunity resources first.
- Improve decisions under prompt truncation risk.

## 51. Ensemble or Voting Mode
- Query multiple action candidates and vote.
- Reduce single-sample model variance.
- Improve robustness on ambiguous steps.

## 52. Self-Reflection Step
- Inject periodic strategy review prompts.
- Summarize progress and missed opportunities.
- Correct weak trajectories mid-episode.

## 53. Environment Seeding for Reproducibility
- Wire deterministic seed into task initialization.
- Make runs repeatable for debugging.
- Improve benchmark comparability.

## 54. Client-Server Integration Tests
- Test via actual FastAPI server and client round-trip.
- Validate serialization and protocol behavior.
- Catch contract-level regressions.

## 55. Config File Support
- Add optional `config.yaml` for runtime parameters.
- Keep env vars as override layer.
- Simplify iterative tuning.

## 56. Action Explanation Logging
- Emit compact per-step reason tags/logs.
- Improve transparency for demos and debugging.
- Keep structured output parser-safe.

## 57. Warm-Start Strategy
- Seed first few steps with known high-confidence moves.
- Hand off remaining steps to adaptive policy.
- Raise baseline stability for known patterns.

## 58. WebSocket Reconnection Handling
- Retry and recover gracefully on dropped session.
- Rehydrate state where protocol allows.
- Prevent full-episode failure from transient disconnects.

## 59. Per-Action-Type Success Metrics
- Track success/failure rates by action type.
- Report stats at episode end.
- Identify weak policy segments quickly.

## 60. Resource Clustering
- Group similar resources into logical clusters.
- Plan actions at cluster level with member expansion.
- Improve scale handling on large tasks.

## 61. Adaptive Heuristic Thresholds
- Make thresholds task-aware and configurable.
- Loosen/tighten based on live progress.
- Improve cross-task performance.

## 62. Dependency Chain Depth Scoring
- Compute transitive dependency depth per resource.
- Penalize actions on deep upstream nodes.
- Improve systemic safety reasoning.

## 63. SLA Recovery Awareness
- Detect SLA violation and switch to safe/stop strategy.
- Avoid unnecessary risky actions after cap-trigger events.
- Preserve remaining score quality.

## 64. Memory-Aware Idle Detection
- Add secondary idle logic for memory-dominant profiles.
- Avoid over-reliance on CPU-only triggers.
- Improve detection of wasteful resources.

## 65. Negative Reward Streak Breaker
- Detect repeated negative-reward streaks.
- Force strategy change or early skip.
- Prevent prolonged penalty spirals.

## 66. Per-Step Timeout Control
- Apply timeout guard around environment step calls.
- Fail fast on hung interactions.
- Improve runtime reliability.

## 67. Savings Percentage in Prompt
- Include progress percentage relative to original cost.
- Make stopping/progress decisions easier.
- Improve interpretability for the model.

## 68. Reserved-to-Spot Upgrade Path
- Allow reserved to spot transition when risk permits.
- Capture additional savings opportunities.
- Use type/criticality gates for safety.

## 69. Structured Error Codes
- Replace free-form errors with machine-readable codes.
- Standardize downstream handling and tests.
- Improve deterministic recovery logic.

## 70. Optimal Path Diff Reporter
- Compare executed sequence against computed best path.
- Report missed opportunities and causes.
- Speed up prompt/policy iteration.

## 71. Combined Idle Score
- Use weighted CPU+memory idle score.
- Replace brittle threshold pairs.
- Improve consistency of idle classification.

## 72. Dynamic System Prompt Injection
- Add lesson snippets from recent outcomes to prompt.
- Reinforce successful safety/optimization behaviors.
- Adapt strategy during long episodes.

## 73. Rate Limit Backoff
- Handle 429 and transient API errors with exponential retry.
- Limit retries and preserve progress.
- Reduce avoidable call failures.

## 74. Resize Upsize Block
- Reject cost-increasing resize proposals in normalization.
- Prevent accidental upsize penalties.
- Keep policy aligned with optimization goals.

## 75. Utilization Trend Hint
- Add trend labels (stable/declining/growing) per resource.
- Improve inference beyond point-in-time averages.
- Reduce short-term misclassification.

## 76. Episode Score Projection
- Log projected final score if agent skips now.
- Guide continue-versus-stop decisions.
- Improve tactical step budgeting.

## 77. Concurrent Environment Sessions
- Run multiple independent task sessions in parallel.
- Reduce wall-clock execution time.
- Require careful output isolation.

## 78. Action History Summarizer
- Build compact summary of completed actions and results.
- Feed concise history back into prompt.
- Preserve context with fewer tokens.

## 79. Edge-Case Test Expansion
- Add tests for corner scenarios and invalid sequences.
- Cover done-state behavior and reset timing.
- Improve correctness confidence.

## 80. OpenAPI Client Auto-Generation
- Generate typed client from server API schema.
- Simplify third-party integration.
- Reduce manual client maintenance risk.

## 81. Multi-Model Fallback Chain
- Route failures through primary then backup model before heuristic.
- Improve robustness against model-specific failures.
- Keep behavior deterministic with clear fallback order.

## 82. Dry-Run Simulation Mode
- Preview expected cost/SLA impact before action commit.
- Reject high-risk low-value actions early.
- Reduce avoidable penalties.

## 83. Prompt Versioning
- Store prompts as versioned files.
- Select active prompt via config/env.
- Enable reproducible prompt experiments.

## 84. LLM Cost Accounting
- Compute per-run inference dollar cost from token usage.
- Compare model spend against cloud savings.
- Flag net-negative optimization runs.

## 85. Human-in-the-Loop Mode
- Add optional interactive approve/override per step.
- Improve demo safety and controllability.
- Help manual debugging during development.

## 86. Resource Tagging System
- Add tag metadata for grouping and policy filters.
- Enable tag-based bulk optimization strategies.
- Improve operational realism.

## 87. Probabilistic SLA Modeling
- Replace deterministic SLA impact with probabilistic effects.
- Model uncertainty in high-risk actions.
- Improve realism and policy robustness.

## 88. Reward Normalization Across Tasks
- Normalize reward scale across easy/medium/hard tasks.
- Reduce policy bias from task-size differences.
- Improve cross-task consistency.

## 89. Transfer Learning Between Tasks
- Carry successful patterns from easier tasks to harder ones.
- Inject learned priors into prompts.
- Improve hard-task startup quality.

## 90. Environment Consistency Validator
- Check invariants after every step.
- Assert cost/state/accounting correctness.
- Detect environment bugs early.

## 91. Two-Phase Commit for Risky Actions
- Require confirmation for high-risk modifications.
- Add safety buffer before irreversible actions.
- Reduce accidental outages.

## 92. Curriculum Learning Mode
- Progressive policy adaptation across difficulty levels.
- Reinforce successful rules stage by stage.
- Improve overall learning efficiency.

## 93. State-Space Coverage Metric
- Track explored versus possible action space.
- Measure search breadth and blind spots.
- Improve exploration strategy tuning.

## 94. Benchmark Leaderboard Export
- Emit standardized run report JSON.
- Include score, steps, latency, token, and model metadata.
- Simplify objective comparisons across variants.

## 95. Task Schema Validation on Load
- Validate task JSON against schema before run.
- Enforce unique IDs and valid dependencies.
- Prevent runtime errors from bad task files.

## 96. Observation Hashing for Change Detection
- Hash observation snapshots each step.
- Detect no-op transitions quickly.
- Avoid redundant reasoning cycles.

## 97. Partial Action Credit
- Adjust reward shaping for larger efficient single-step improvements.
- Incentivize action efficiency.
- Reduce reward fragmentation artifacts.

## 98. CI/CD Pipeline
- Add automated lint, type-check, tests, and smoke runs.
- Run on push/pull request.
- Protect repository stability.

## 99. Server Health Probe Before Episode
- Verify server readiness with retry/backoff before reset.
- Avoid immediate crash on unavailable endpoint.
- Improve startup robustness.

## 100. Dockerized End-to-End Test
- Add compose-based integration run with mock/stub LLM.
- Validate complete deployment workflow.
- Ensure reproducible release checks.
