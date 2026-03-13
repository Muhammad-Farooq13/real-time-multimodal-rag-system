from fastapi.testclient import TestClient

from app.main import app


def test_metrics_endpoints_expose_request_data() -> None:
    client = TestClient(app)

    query_resp = client.post(
        "/v1/query",
        json={
            "mode": "text",
            "text": "What is retrieval augmented generation?",
            "top_k": 3,
        },
    )
    assert query_resp.status_code == 200
    assert query_resp.headers["X-Trace-Id"]
    assert query_resp.headers["X-Cache-Status"] in {"miss", "exact_hit", "semantic_hit"}

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    body = metrics_resp.json()
    assert body["counters"]["requests_total"] >= 1
    assert body["latency"]["count"] >= 1

    prom_resp = client.get("/metrics/prometheus")
    assert prom_resp.status_code == 200
    assert "rag_requests_total" in prom_resp.text
    assert "rag_cache_events_total" in prom_resp.text
