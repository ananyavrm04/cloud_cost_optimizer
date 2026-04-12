"""
Run inference.py across multiple models and print a compact comparison table.

Example:
    python scripts/compare_models.py --models openai/gpt-oss-20b:fastest openai/gpt-oss-120b:fastest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_once(model_name: str) -> dict:
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    completed = subprocess.run(
        [sys.executable, "inference.py"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    report_path = Path("artifacts/benchmark_report.json")
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    return {
        "model": model_name,
        "exit_code": completed.returncode,
        "scores": report.get("scores", {}),
        "llm_calls": int(report.get("llm_usage", {}).get("calls", 0)),
        "llm_total_tokens": int(report.get("llm_usage", {}).get("total_tokens", 0)),
        "force_llm_every_step": bool(report.get("force_llm_every_step", False)),
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-6:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-6:]),
    }


def _score_avg(scores: dict) -> float:
    try:
        return (
            float(scores.get("easy", 0.0))
            + float(scores.get("medium", 0.0))
            + float(scores.get("hard", 0.0))
        ) / 3.0
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare inference runs across model names.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model ids to test, e.g. openai/gpt-oss-20b:fastest openai/gpt-oss-120b:fastest",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for model in args.models:
        print(f"[RUN] model={model}", flush=True)
        rows.append(run_once(model))

    print("\nModel Comparison")
    print("model | exit | easy | medium | hard | avg | llm_calls | total_tokens | force_llm")
    print("-" * 100)
    for r in rows:
        s = r["scores"]
        print(
            f"{r['model']} | {r['exit_code']} | "
            f"{float(s.get('easy', 0.0)):.4f} | {float(s.get('medium', 0.0)):.4f} | {float(s.get('hard', 0.0)):.4f} | "
            f"{_score_avg(s):.4f} | {r['llm_calls']} | {r['llm_total_tokens']} | {int(r['force_llm_every_step'])}"
        )

    out = Path("artifacts/model_compare_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out}")


if __name__ == "__main__":
    main()

