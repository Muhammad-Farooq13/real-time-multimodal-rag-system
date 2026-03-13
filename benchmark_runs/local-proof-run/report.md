# Benchmark Report: local-proof-run

Generated: 2026-03-13T05:52:40.957846+00:00

## Run Metadata

- Base URL: http://localhost:8000
- Started: 2026-03-13T05:52:20.002054+00:00
- Finished: 2026-03-13T05:52:40.957833+00:00
- Duration seconds: 20.96

## k6 Summary

- HTTP requests: 301
- Iterations: 301
- Max VUs observed: 10
- Failure rate: 0.0000
- HTTP latency avg: 4.29 ms
- HTTP latency p50: 3.52 ms
- HTTP latency p95: 4.93 ms
- HTTP latency p99: 66.74 ms

## Application Metrics Delta

- Request count delta: 301
- Exact cache hit delta: 301
- Semantic cache hit delta: 0
- Cache miss delta: 0
- Cache hit ratio: 100.00%
- App latency p95 before: 0.98 ms
- App latency p95 after: 1.11 ms
- App latency p99 after: 1.29 ms

## Evidence Files

- k6 summary: k6_summary.json
- Metrics before: metrics_before.json
- Metrics after: metrics_after.json

## Interpretation

- Use the k6 p95 and p99 values to assess external client-perceived latency.
- Use the app metrics deltas to explain whether performance gains came from exact cache reuse, semantic reuse, or retrieval path improvements.
- Commit this report alongside the raw JSON outputs to make performance claims reproducible.
