from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestTrace:
    trace_id: str
    mode: str
    vector_backend: str
    fast_mode: bool
    cache_status: str = "miss"
    retrieved_docs: int = 0
    returned_citations: int = 0
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def add_stage_timing(self, stage: str, duration_ms: float) -> None:
        self.stage_timings_ms[stage] = round(float(duration_ms), 4)

    def to_log_dict(self) -> dict:
        total_ms = round(sum(self.stage_timings_ms.values()), 4)
        return {
            "event": "rag_request_trace",
            "trace_id": self.trace_id,
            "mode": self.mode,
            "vector_backend": self.vector_backend,
            "fast_mode": self.fast_mode,
            "cache_status": self.cache_status,
            "retrieved_docs": self.retrieved_docs,
            "returned_citations": self.returned_citations,
            "stage_timings_ms": self.stage_timings_ms,
            "total_pipeline_ms": total_ms,
        }
