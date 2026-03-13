from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from app.services.benchmarking import load_json, parse_k6_summary, render_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k6 benchmark and generate a performance report")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--script", type=Path, default=Path("benchmarks/k6_query.js"))
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--duration", type=str, default="1m")
    parser.add_argument("--preallocated-vus", type=int, default=50)
    parser.add_argument("--max-vus", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_runs"))
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def fetch_metrics(base_url: str) -> dict:
    with httpx.Client(timeout=15.0) as client:
        response = client.get(f"{base_url.rstrip('/')}/metrics")
        response.raise_for_status()
        return response.json()


def ensure_k6_available() -> str:
    explicit = os.environ.get("K6_PATH", "").strip()
    if explicit and Path(explicit).exists():
        return explicit

    path = shutil.which("k6")
    if path is not None:
        return path

    default_windows_path = Path("C:/Program Files/k6/k6.exe")
    if default_windows_path.exists():
        return str(default_windows_path)

    raise SystemExit("k6 was not found on PATH. Install k6 or set K6_PATH.")
    return path


def main() -> None:
    args = parse_args()
    k6_path = ensure_k6_available()

    started_at = datetime.now(timezone.utc)
    run_name = args.run_name or started_at.strftime("run-%Y%m%d-%H%M%S")
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_before = fetch_metrics(args.base_url)
    metrics_before_path = run_dir / "metrics_before.json"
    metrics_before_path.write_text(__import__("json").dumps(metrics_before, indent=2), encoding="utf-8")

    k6_summary_path = run_dir / "k6_summary.json"
    cmd = [
        k6_path,
        "run",
        str(args.script),
        "--summary-export",
        str(k6_summary_path),
    ]
    env = {
        **__import__("os").environ,
        "BASE_URL": args.base_url,
        "RATE": str(args.rate),
        "DURATION": args.duration,
        "PREALLOCATED_VUS": str(args.preallocated_vus),
        "MAX_VUS": str(args.max_vus),
    }

    print(f"Running benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    metrics_after = fetch_metrics(args.base_url)
    metrics_after_path = run_dir / "metrics_after.json"
    metrics_after_path.write_text(__import__("json").dumps(metrics_after, indent=2), encoding="utf-8")

    k6_summary = load_json(k6_summary_path)
    k6_metrics = parse_k6_summary(k6_summary)

    finished_at = datetime.now(timezone.utc)
    report_text = render_benchmark_report(
        run_name=run_name,
        base_url=args.base_url,
        started_at=started_at,
        finished_at=finished_at,
        k6_metrics=k6_metrics,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        k6_summary_path=k6_summary_path,
        metrics_before_path=metrics_before_path,
        metrics_after_path=metrics_after_path,
    )
    report_path = run_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Benchmark artifacts written to {run_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
