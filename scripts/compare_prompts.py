"""
Run inference with multiple prompt versions and print score comparison.

Usage:
    python scripts/compare_prompts.py
    python scripts/compare_prompts.py v1 v2
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Dict, List


SUMMARY_RE = re.compile(
    r"\[SUMMARY\]\s+easy=([0-9.]+)\s+medium=([0-9.]+)\s+hard=([0-9.]+)"
)


def run_for_prompt(prompt_version: str) -> Dict[str, float]:
    env = os.environ.copy()
    env["PROMPT_VERSION"] = prompt_version
    proc = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Prompt {prompt_version} failed (exit={proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    match = SUMMARY_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(
            f"Prompt {prompt_version} missing [SUMMARY] line.\nSTDOUT:\n{proc.stdout}"
        )
    easy, medium, hard = map(float, match.groups())
    return {"easy": easy, "medium": medium, "hard": hard, "avg": (easy + medium + hard) / 3.0}


def main(argv: List[str]) -> int:
    versions = argv if argv else ["v1", "v2"]
    print("Prompt comparison:")
    print("-" * 74)
    print(f"{'Prompt':<12} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Average':>10}")
    print("-" * 74)
    rows = []
    for v in versions:
        scores = run_for_prompt(v)
        rows.append((v, scores))
        print(
            f"{v:<12} {scores['easy']:>8.4f} {scores['medium']:>8.4f} "
            f"{scores['hard']:>8.4f} {scores['avg']:>10.4f}"
        )
    best = max(rows, key=lambda x: x[1]["avg"])
    print("-" * 74)
    print(f"Best prompt by average score: {best[0]} ({best[1]['avg']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
