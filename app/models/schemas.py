from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    env: str


class Citation(BaseModel):
    doc_id: str
    score: float
    snippet: str


class QueryRequest(BaseModel):
    mode: Literal["text", "image", "audio"]
    text: str | None = None
    image_b64: str | None = None
    audio_b64: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    max_context_chunks: int = Field(default=6, ge=1, le=20)
    use_hybrid: bool = True
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    use_rerank: bool = False
    rerank_top_n: int = Field(default=10, ge=1, le=50)
    fast_mode: bool = True

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "QueryRequest":
        if self.mode == "text" and not self.text:
            raise ValueError("text is required when mode='text'")
        if self.mode == "image" and not self.image_b64:
            raise ValueError("image_b64 is required when mode='image'")
        if self.mode == "audio" and not self.audio_b64:
            raise ValueError("audio_b64 is required when mode='audio'")
        return self


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: float
    mode: str
    trace_id: str
