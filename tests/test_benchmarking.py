from datetime import datetime, timezone
from pathlib import Path

from app.services.benchmarking import parse_k6_summary, render_benchmark_report


def test_parse_k6_summary_extracts_key_metrics() -> None:
    summary = {
        "metrics": {
            "http_reqs": {"values": {"count": 1200}},
            "http_req_failed": {"values": {"rate": 0.0}},
            "http_req_duration": {"values": {"avg": 32.1, "p(50)": 28.0, "p(95)": 58.2, "p(99)": 71.4}},
            "iteration_duration": {"values": {"avg": 35.6}},
            "iterations": {"values": {"count": 1200}},
            "vus_max": {"values": {"max": 75}},
        }
    }

    parsed = parse_k6_summary(summary)
    assert parsed["http_reqs"] == 1200.0
    assert parsed["http_req_duration_p95"] == 58.2
    assert parsed["vus_max"] == 75.0


def test_render_benchmark_report_contains_key_sections() -> None:
    report = render_benchmark_report(
        run_name="test-run",
        base_url="http://localhost:8000",
        started_at=datetime(2026, 3, 13, 12, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 3, 13, 12, 1, tzinfo=timezone.utc),
        k6_metrics={
            "http_reqs": 1000.0,
            "iterations": 1000.0,
            "vus_max": 60.0,
            "http_req_failed_rate": 0.0,
            "http_req_duration_avg": 30.0,
            "http_req_duration_p50": 25.0,
            "http_req_duration_p95": 55.0,
            "http_req_duration_p99": 70.0,
            "iteration_duration_avg": 32.0,
        },
        metrics_before={
            "counters": {"requests_total": 0.0, "cache_exact_hit_total": 0.0, "cache_semantic_hit_total": 0.0, "cache_miss_total": 0.0},
            "latency": {"p95_ms": 0.0, "p99_ms": 0.0},
        },
        metrics_after={
            "counters": {"requests_total": 1000.0, "cache_exact_hit_total": 300.0, "cache_semantic_hit_total": 100.0, "cache_miss_total": 600.0},
            "latency": {"p95_ms": 45.0, "p99_ms": 68.0},
        },
        k6_summary_path=Path("k6_summary.json"),
        metrics_before_path=Path("metrics_before.json"),
        metrics_after_path=Path("metrics_after.json"),
    )

    assert "# Benchmark Report: test-run" in report
    assert "Cache hit ratio: 40.00%" in report
    assert "HTTP latency p95: 55.00 ms" in report
