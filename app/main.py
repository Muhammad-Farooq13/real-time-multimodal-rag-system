from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse

from app.core.config import settings
from app.models.schemas import HealthResponse, QueryRequest, QueryResponse
from app.services.metrics import metrics_registry
from app.services.rag_pipeline import RagPipeline

app = FastAPI(title=settings.app_name, version="0.1.0")
pipeline = RagPipeline()
logger = logging.getLogger("rag.service")

if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, env=settings.app_env)


@app.get("/metrics")
def metrics() -> dict:
    return metrics_registry.snapshot()


@app.get("/metrics/prometheus", response_class=PlainTextResponse)
def metrics_prometheus() -> str:
    return metrics_registry.to_prometheus_text()


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest, response: Response) -> QueryResponse:
    if request.top_k > settings.max_top_k:
        raise HTTPException(status_code=400, detail=f"top_k cannot exceed {settings.max_top_k}")

    metrics_registry.incr("requests_total")
    metrics_registry.incr(f"requests_mode_{request.mode}_total")

    trace_id = str(uuid.uuid4())
    started = time.perf_counter()

    answer, citations, trace = pipeline.answer(request, trace_id=trace_id)

    latency_ms = (time.perf_counter() - started) * 1000
    metrics_registry.observe_latency(latency_ms)
    trace.add_stage_timing("request_total", latency_ms)
    logger.info(json.dumps(trace.to_log_dict(), ensure_ascii=True))

    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Cache-Status"] = trace.cache_status
    return QueryResponse(
        answer=answer,
        citations=citations,
        latency_ms=round(latency_ms, 2),
        mode=request.mode,
        trace_id=trace_id,
    )
