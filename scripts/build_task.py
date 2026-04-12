"""
Custom task builder CLI for Cloud Cost Optimizer.

Generates validator-compatible task JSON files with configurable difficulty knobs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SIZE_MULTIPLIERS = {
    "small": 1.0,
    "medium": 2.0,
    "large": 4.0,
    "xlarge": 8.0,
}

TYPE_WEIGHTS = [
    ("compute", 0.75),
    ("storage", 0.15),
    ("database", 0.10),
]


def weighted_pick(rng: random.Random) -> str:
    v = rng.random()
    acc = 0.0
    for t, w in TYPE_WEIGHTS:
        acc += w
        if v <= acc:
            return t
    return "compute"


def cost_for_size(size: str) -> float:
    return 120.0 * SIZE_MULTIPLIERS[size]


def build_resource(
    *,
    idx: int,
    role: str,
    r_type: str,
    size: str,
    cpu: float,
    mem: float,
    is_critical: bool,
    eligible_reserved: bool,
) -> dict:
    rid = f"{role}-{r_type[:2]}-{idx}"
    return {
        "id": rid,
        "name": f"{role.title()} {r_type.title()} {idx}",
        "type": r_type,
        "size": size,
        "cpu_usage_avg": round(cpu, 1),
        "mem_usage_avg": round(mem, 1),
        "cost_per_month": round(cost_for_size(size), 2),
        "is_critical": is_critical,
        "dependencies": [],
        "pricing": "on_demand",
        "eligible_for_reserved": eligible_reserved,
        "status": "running",
    }


def estimate_optimal_savings(resources: list[dict]) -> float:
    total = 0.0
    for r in resources:
        if r["is_critical"]:
            continue
        # Idle non-critical -> terminate candidate.
        if r["cpu_usage_avg"] < 6.0 and r["mem_usage_avg"] < 12.0 and not r["dependencies"]:
            total += r["cost_per_month"]
            continue
        # Oversized-ish -> one-step downsize estimate.
        if r["size"] in {"xlarge", "large"} and r["cpu_usage_avg"] < 35.0:
            current = SIZE_MULTIPLIERS[r["size"]]
            next_size = "large" if r["size"] == "xlarge" else "medium"
            nxt = SIZE_MULTIPLIERS[next_size]
            total += r["cost_per_month"] * (1.0 - (nxt / current))
        # Reserved candidate estimate.
        if r["eligible_for_reserved"] and r["pricing"] == "on_demand":
            total += r["cost_per_month"] * 0.4
    return round(max(0.0, total), 2)


def add_dependencies(
    resources: list[dict],
    *,
    rng: random.Random,
    max_depth: int,
    extra_edges: int,
) -> None:
    if max_depth <= 0:
        return

    n = len(resources)
    levels = [rng.randint(0, max_depth) for _ in range(n)]

    # Build at least one critical-ish chain if possible.
    chain = [i for i, r in enumerate(resources) if r["is_critical"]][: max_depth + 1]
    for i in range(1, len(chain)):
        child = resources[chain[i]]
        parent = resources[chain[i - 1]]
        child["dependencies"].append(parent["id"])
        levels[chain[i]] = min(max_depth, levels[chain[i - 1]] + 1)

    # Add extra random acyclic edges (depend only on lower/equal level-1).
    possible = []
    for i in range(n):
        for j in range(i):
            if levels[j] <= levels[i] - 1:
                possible.append((i, j))

    rng.shuffle(possible)
    used = 0
    for i, j in possible:
        if used >= extra_edges:
            break
        child = resources[i]
        parent_id = resources[j]["id"]
        if parent_id in child["dependencies"]:
            continue
        child["dependencies"].append(parent_id)
        used += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build custom Cloud Cost Optimizer task JSON.")
    parser.add_argument("--task-id", default="custom", help="Task id name (used in output filename if no --output).")
    parser.add_argument("--resources", type=int, default=30, help="Total resource count.")
    parser.add_argument("--critical-ratio", type=float, default=0.2, help="Fraction of critical resources (0..1).")
    parser.add_argument("--idle-ratio", type=float, default=0.25, help="Fraction of idle resources (0..1).")
    parser.add_argument("--oversized-ratio", type=float, default=0.25, help="Fraction of oversized resources (0..1).")
    parser.add_argument("--reserved-eligible-ratio", type=float, default=0.25, help="Fraction eligible for reserved/spot.")
    parser.add_argument("--dependency-depth", type=int, default=3, help="Maximum dependency chain depth.")
    parser.add_argument("--dependency-edges", type=int, default=10, help="Extra dependency edges to add.")
    parser.add_argument("--sla-target", type=float, default=99.9, help="SLA target uptime.")
    parser.add_argument("--initial-uptime", type=float, default=100.0, help="Starting uptime.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", default="", help="Output path. Default: tasks/<task-id>.json")
    args = parser.parse_args()

    if args.resources <= 0:
        raise ValueError("--resources must be > 0")
    if not (0.0 <= args.critical_ratio <= 1.0):
        raise ValueError("--critical-ratio must be within [0,1]")
    if not (0.0 <= args.idle_ratio <= 1.0):
        raise ValueError("--idle-ratio must be within [0,1]")
    if not (0.0 <= args.oversized_ratio <= 1.0):
        raise ValueError("--oversized-ratio must be within [0,1]")
    if not (0.0 <= args.reserved_eligible_ratio <= 1.0):
        raise ValueError("--reserved-eligible-ratio must be within [0,1]")

    rng = random.Random(args.seed)

    n = args.resources
    n_critical = int(round(n * args.critical_ratio))
    n_idle = int(round(n * args.idle_ratio))
    n_oversized = int(round(n * args.oversized_ratio))
    n_reserved = int(round(n * args.reserved_eligible_ratio))

    resources: list[dict] = []
    for i in range(1, n + 1):
        role = "active"
        size = "large"
        cpu, mem = (rng.uniform(40, 75), rng.uniform(35, 80))

        if i <= n_idle:
            role = "idle"
            size = "large"
            cpu, mem = (rng.uniform(1, 5), rng.uniform(3, 10))
        elif i <= n_idle + n_oversized:
            role = "oversized"
            size = "xlarge"
            cpu, mem = (rng.uniform(15, 32), rng.uniform(14, 34))

        r_type = weighted_pick(rng)
        is_critical = i <= n_critical
        eligible_reserved = i <= n_reserved

        # Keep some type-aware realism.
        if r_type == "storage" and role == "oversized":
            cpu, mem = (rng.uniform(8, 20), rng.uniform(10, 28))
        if r_type == "database" and role == "idle":
            cpu, mem = (rng.uniform(1, 6), rng.uniform(4, 14))

        resources.append(
            build_resource(
                idx=i,
                role=role,
                r_type=r_type,
                size=size,
                cpu=cpu,
                mem=mem,
                is_critical=is_critical,
                eligible_reserved=eligible_reserved,
            )
        )

    add_dependencies(
        resources,
        rng=rng,
        max_depth=max(0, args.dependency_depth),
        extra_edges=max(0, args.dependency_edges),
    )

    task = {
        "resources": resources,
        "optimal_savings": estimate_optimal_savings(resources),
        "sla_target": float(args.sla_target),
        "initial_uptime": float(args.initial_uptime),
    }

    output = Path(args.output) if args.output else Path("tasks") / f"{args.task_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "task_id": args.task_id,
                "output": str(output),
                "resources": n,
                "critical": n_critical,
                "idle": n_idle,
                "oversized": n_oversized,
                "reserved_eligible": n_reserved,
                "estimated_optimal_savings": task["optimal_savings"],
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

