"""
Enhancement #8: local pre-submit validator.

Checks:
- required env vars (submission mode)
- inference stdout contains [START]/[STEP]/[END]
- scores are strictly within (0, 1)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    required = ["API_BASE_URL", "API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        return fail(f"Missing required env vars: {', '.join(missing)}")

    proc = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout
    stderr = proc.stderr

    if proc.returncode != 0:
        print(stdout)
        print(stderr)
        return fail(f"inference.py exited with code {proc.returncode}")

    if "[START]" not in stdout or "[STEP]" not in stdout or "[END]" not in stdout:
        return fail("Structured output missing one of [START]/[STEP]/[END]")
    if "[SUMMARY]" not in stdout:
        return fail("Structured output missing [SUMMARY] line")

    starts = re.findall(r"^\[START\]\s+task=(\w+)\s*$", stdout, flags=re.MULTILINE)
    ends = re.findall(r"^\[END\]\s+task=(\w+)\s+score=([0-9.]+)\s+steps=\d+\s*$", stdout, flags=re.MULTILINE)
    if len(starts) != 3 or len(ends) != 3:
        return fail("Expected exactly 3 [START] and 3 [END] lines")

    end_scores = re.findall(r"\[END\]\s+task=\w+\s+score=([0-9.]+)", stdout)
    if len(end_scores) != 3:
        return fail("Expected exactly 3 [END] score lines")

    for s in end_scores:
        try:
            value = float(s)
        except ValueError:
            return fail(f"Invalid numeric score: {s}")
        if not (0.0 < value < 1.0):
            return fail(f"Score out of range (0,1): {value}")

    benchmark_path = Path("artifacts/benchmark_report.json")
    if not benchmark_path.exists():
        return fail("Missing artifacts/benchmark_report.json after run")

    print("[OK] pre-submit checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
