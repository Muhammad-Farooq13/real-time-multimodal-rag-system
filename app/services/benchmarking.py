from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(summary: dict, metric_name: str, field: str, default: float = 0.0) -> float:
    metric = summary.get("metrics", {}).get(metric_name, {})
    values = metric.get("values")
    if isinstance(values, dict) and field in values:
        raw = values.get(field, default)
    else:
        raw = metric.get(field, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_k6_summary(summary: dict) -> dict:
    p50 = _metric_value(summary, "http_req_duration", "p(50)")
    if p50 == 0.0:
        p50 = _metric_value(summary, "http_req_duration", "med")

    p99 = _metric_value(summary, "http_req_duration", "p(99)")
    if p99 == 0.0:
        p99 = _metric_value(summary, "http_req_duration", "max")

    return {
        "vus_max": _metric_value(summary, "vus_max", "max"),
        "http_reqs": _metric_value(summary, "http_reqs", "count"),
        "http_req_failed_rate": _metric_value(summary, "http_req_failed", "rate")
        or _metric_value(summary, "http_req_failed", "value"),
        "http_req_duration_avg": _metric_value(summary, "http_req_duration", "avg"),
        "http_req_duration_p50": p50,
        "http_req_duration_p95": _metric_value(summary, "http_req_duration", "p(95)"),
        "http_req_duration_p99": p99,
        "iteration_duration_avg": _metric_value(summary, "iteration_duration", "avg"),
        "iterations": _metric_value(summary, "iterations", "count"),
    }


def render_benchmark_report(
    *,
    run_name: str,
    base_url: str,
    started_at: datetime,
    finished_at: datetime,
    k6_metrics: dict,
    metrics_before: dict,
    metrics_after: dict,
    k6_summary_path: Path,
    metrics_before_path: Path,
    metrics_after_path: Path,
) -> str:
    before_counters = metrics_before.get("counters", {})
    after_counters = metrics_after.get("counters", {})
    before_latency = metrics_before.get("latency", {})
    after_latency = metrics_after.get("latency", {})

    cache_exact_delta = after_counters.get("cache_exact_hit_total", 0.0) - before_counters.get(
        "cache_exact_hit_total", 0.0
    )
    cache_semantic_delta = after_counters.get(
        "cache_semantic_hit_total", 0.0
    ) - before_counters.get("cache_semantic_hit_total", 0.0)
    cache_miss_delta = after_counters.get("cache_miss_total", 0.0) - before_counters.get(
        "cache_miss_total", 0.0
    )

    total_cache_events = max(cache_exact_delta + cache_semantic_delta + cache_miss_delta, 1.0)
    cache_hit_ratio = (cache_exact_delta + cache_semantic_delta) / total_cache_events

    return f"""# Benchmark Report: {run_name}

Generated: {datetime.now(timezone.utc).isoformat()}

## Run Metadata

- Base URL: {base_url}
- Started: {started_at.isoformat()}
- Finished: {finished_at.isoformat()}
- Duration seconds: {(finished_at - started_at).total_seconds():.2f}

## k6 Summary

- HTTP requests: {k6_metrics['http_reqs']:.0f}
- Iterations: {k6_metrics['iterations']:.0f}
- Max VUs observed: {k6_metrics['vus_max']:.0f}
- Failure rate: {k6_metrics['http_req_failed_rate']:.4f}
- HTTP latency avg: {k6_metrics['http_req_duration_avg']:.2f} ms
- HTTP latency p50: {k6_metrics['http_req_duration_p50']:.2f} ms
- HTTP latency p95: {k6_metrics['http_req_duration_p95']:.2f} ms
- HTTP latency p99: {k6_metrics['http_req_duration_p99']:.2f} ms

## Application Metrics Delta

- Request count delta: {after_counters.get('requests_total', 0.0) - before_counters.get('requests_total', 0.0):.0f}
- Exact cache hit delta: {cache_exact_delta:.0f}
- Semantic cache hit delta: {cache_semantic_delta:.0f}
- Cache miss delta: {cache_miss_delta:.0f}
- Cache hit ratio: {cache_hit_ratio:.2%}
- App latency p95 before: {before_latency.get('p95_ms', 0.0):.2f} ms
- App latency p95 after: {after_latency.get('p95_ms', 0.0):.2f} ms
- App latency p99 after: {after_latency.get('p99_ms', 0.0):.2f} ms

## Evidence Files

- k6 summary: {k6_summary_path.name}
- Metrics before: {metrics_before_path.name}
- Metrics after: {metrics_after_path.name}

## Interpretation

- Use the k6 p95 and p99 values to assess external client-perceived latency.
- Use the app metrics deltas to explain whether performance gains came from exact cache reuse, semantic reuse, or retrieval path improvements.
- Commit this report alongside the raw JSON outputs to make performance claims reproducible.
"""
