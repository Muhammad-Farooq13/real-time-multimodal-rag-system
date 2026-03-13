from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._latencies_ms: list[float] = []

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(float(latency_ms))
            if len(self._latencies_ms) > 10000:
                self._latencies_ms = self._latencies_ms[-10000:]

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self._latencies_ms)
            counters = dict(self._counters)

        total = len(latencies)
        if total == 0:
            return {
                "counters": counters,
                "latency": {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0},
            }

        ordered = sorted(latencies)
        return {
            "counters": counters,
            "latency": {
                "count": total,
                "avg_ms": round(sum(ordered) / total, 4),
                "p50_ms": round(self._percentile(ordered, 0.50), 4),
                "p95_ms": round(self._percentile(ordered, 0.95), 4),
                "p99_ms": round(self._percentile(ordered, 0.99), 4),
            },
        }

    def to_prometheus_text(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP rag_requests_total Total query requests processed.",
            "# TYPE rag_requests_total counter",
            f"rag_requests_total {snapshot['counters'].get('requests_total', 0.0)}",
            "# HELP rag_request_latency_ms Request latency summary in milliseconds.",
            "# TYPE rag_request_latency_ms gauge",
            f"rag_request_latency_ms_avg {snapshot['latency']['avg_ms']}",
            f"rag_request_latency_ms_p50 {snapshot['latency']['p50_ms']}",
            f"rag_request_latency_ms_p95 {snapshot['latency']['p95_ms']}",
            f"rag_request_latency_ms_p99 {snapshot['latency']['p99_ms']}",
            "# HELP rag_cache_events_total Cache hit and miss counters.",
            "# TYPE rag_cache_events_total counter",
            f"rag_cache_events_total{{type=\"exact_hit\"}} {snapshot['counters'].get('cache_exact_hit_total', 0.0)}",
            f"rag_cache_events_total{{type=\"semantic_hit\"}} {snapshot['counters'].get('cache_semantic_hit_total', 0.0)}",
            f"rag_cache_events_total{{type=\"miss\"}} {snapshot['counters'].get('cache_miss_total', 0.0)}",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight


metrics_registry = MetricsRegistry()
