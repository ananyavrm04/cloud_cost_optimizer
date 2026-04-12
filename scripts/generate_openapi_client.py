"""
Enhancement #80: Generate a typed client stub from the server's OpenAPI schema.

Usage:
    python scripts/generate_openapi_client.py
    python scripts/generate_openapi_client.py --url http://localhost:7860 --output generated_client.py
"""

from __future__ import annotations

import argparse
import json
import sys

import requests


def fetch_schema(base_url: str) -> dict:
    resp = requests.get(f"{base_url.rstrip('/')}/openapi.json", timeout=10)
    resp.raise_for_status()
    return resp.json()


def generate_stub(schema: dict) -> str:
    lines = [
        '"""Auto-generated typed client stub from OpenAPI schema."""',
        "",
        "from typing import Any, Dict",
        "",
        "import requests",
        "",
        "",
        "class CloudCostOptimizerClient:",
        '    """Typed HTTP client for Cloud Cost Optimizer API."""',
        "",
        "    def __init__(self, base_url: str):",
        "        self.base_url = base_url.rstrip('/')",
        "",
    ]
    paths = schema.get("paths", {})
    for path, methods in paths.items():
        for method, details in methods.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            op_id = details.get("operationId", f"{method}_{path}")
            func_name = op_id.replace("/", "_").replace("-", "_").replace(".", "_").strip("_")
            summary = details.get("summary", "")
            lines.append(f"    def {func_name}(self, **kwargs) -> Dict[str, Any]:")
            lines.append(f'        """{summary}"""')
            lines.append(f'        resp = requests.{method}(f"{{self.base_url}}{path}", **kwargs)')
            lines.append("        resp.raise_for_status()")
            lines.append("        return resp.json()")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate typed client from OpenAPI schema")
    parser.add_argument("--url", default="http://localhost:7860", help="Server base URL")
    parser.add_argument("--output", default="generated_client.py", help="Output file path")
    args = parser.parse_args()

    try:
        schema = fetch_schema(args.url)
    except Exception as exc:
        print(f"[FAIL] Could not fetch schema from {args.url}: {exc}")
        return 1

    stub = generate_stub(schema)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(stub)
    print(f"[OK] Generated client stub: {args.output}")
    print(f"     Endpoints: {len(schema.get('paths', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
