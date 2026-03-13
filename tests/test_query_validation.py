from fastapi.testclient import TestClient

from app.main import app


def test_query_requires_text_for_text_mode() -> None:
    client = TestClient(app)
    resp = client.post("/v1/query", json={"mode": "text", "top_k": 3})
    assert resp.status_code == 422
