from __future__ import annotations

import base64
import json
from pathlib import Path
from time import perf_counter

from app.models.schemas import Citation, QueryRequest
from app.services.cache import ResponseCache
from app.services.embeddings import Embedder
from app.services.generator import ResponseGenerator
from app.services.metrics import metrics_registry
from app.services.reranker import Reranker
from app.services.retrieval import HybridRetriever
from app.services.tracing import RequestTrace
from app.services.vector_store import create_vector_store


class RagPipeline:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.cache = ResponseCache()
        self.store = create_vector_store()
        self.generator = ResponseGenerator()
        self.reranker = Reranker()
        self.corpus_rows: list[dict] = []
        self.hybrid_retriever: HybridRetriever | None = None
        self._load_default_index()

    @property
    def vector_backend_name(self) -> str:
        return type(self.store).__name__

    def _load_default_index(self) -> None:
        index_path = Path("data/processed/index.json")
        if not index_path.exists():
            return

        rows = json.loads(index_path.read_text(encoding="utf-8"))
        self.corpus_rows = rows
        self.hybrid_retriever = HybridRetriever(rows) if rows else None
        for row in rows:
            self.store.add(vector=self.embedder.embed(row["text"]), payload=row)

    def _normalize_query_text(self, request: QueryRequest) -> str:
        if request.mode == "text":
            return request.text or ""
        if request.mode == "image":
            # Placeholder for vision encoder extraction.
            size = len(base64.b64decode(request.image_b64 or "", validate=False))
            return f"image-query bytes={size}"
        # Placeholder for ASR output from audio.
        size = len(base64.b64decode(request.audio_b64 or "", validate=False))
        return f"audio-query bytes={size}"

    def answer(self, request: QueryRequest, trace_id: str) -> tuple[str, list[Citation], RequestTrace]:
        trace = RequestTrace(
            trace_id=trace_id,
            mode=request.mode,
            vector_backend=self.vector_backend_name,
            fast_mode=request.fast_mode,
        )

        stage_started = perf_counter()
        query_text = self._normalize_query_text(request)
        trace.add_stage_timing("normalize", (perf_counter() - stage_started) * 1000)

        stage_started = perf_counter()
        qvec = self.embedder.embed(query_text)
        trace.add_stage_timing("embed", (perf_counter() - stage_started) * 1000)

        params_key = self.cache.make_params_key(
            {
                "top_k": request.top_k,
                "max_context_chunks": request.max_context_chunks,
                "use_hybrid": request.use_hybrid,
                "hybrid_alpha": request.hybrid_alpha,
                "use_rerank": request.use_rerank,
                "rerank_top_n": request.rerank_top_n,
                "fast_mode": request.fast_mode,
            }
        )
        cache_key = self.cache.make_cache_key(request.mode, params_key, query_text)

        stage_started = perf_counter()
        exact_cached = self.cache.get_exact(cache_key)
        trace.add_stage_timing("cache_exact_lookup", (perf_counter() - stage_started) * 1000)
        if exact_cached is not None:
            metrics_registry.incr("cache_exact_hit_total")
            trace.cache_status = "exact_hit"
            citations = [Citation(**item) for item in exact_cached.get("citations", [])]
            trace.returned_citations = len(citations)
            return str(exact_cached.get("answer", "")), citations, trace

        stage_started = perf_counter()
        semantic_cached = self.cache.get_semantic(request.mode, params_key, qvec)
        trace.add_stage_timing("cache_semantic_lookup", (perf_counter() - stage_started) * 1000)
        if semantic_cached is not None:
            metrics_registry.incr("cache_semantic_hit_total")
            trace.cache_status = "semantic_hit"
            citations = [Citation(**item) for item in semantic_cached.get("citations", [])]
            trace.returned_citations = len(citations)
            return str(semantic_cached.get("answer", "")), citations, trace

        metrics_registry.incr("cache_miss_total")

        dense_k = request.top_k * 3 if request.use_hybrid else request.top_k
        stage_started = perf_counter()
        hits = self.store.search(query_vector=qvec, top_k=dense_k)
        trace.add_stage_timing("retrieve_dense", (perf_counter() - stage_started) * 1000)

        if request.use_hybrid and self.hybrid_retriever is not None:
            stage_started = perf_counter()
            hits = self.hybrid_retriever.blend(
                query_text=query_text,
                dense_hits=hits,
                top_k=request.top_k,
                alpha=request.hybrid_alpha,
            )
            trace.add_stage_timing("retrieve_hybrid", (perf_counter() - stage_started) * 1000)
        else:
            hits = hits[: request.top_k]

        if request.use_rerank and hits:
            stage_started = perf_counter()
            hits = self.reranker.rerank(
                query_text=query_text,
                candidates=hits,
                top_n=min(request.rerank_top_n, len(hits)),
                embedder=self.embedder,
            )
            trace.add_stage_timing("rerank", (perf_counter() - stage_started) * 1000)

        trace.retrieved_docs = len(hits)

        stage_started = perf_counter()
        answer = self.generator.generate(
            query_text=query_text,
            contexts=hits[: request.max_context_chunks],
            fast_mode=request.fast_mode,
        )
        trace.add_stage_timing("generate", (perf_counter() - stage_started) * 1000)
        citations = [
            Citation(doc_id=h.doc_id, score=round(h.score, 4), snippet=h.text[:240]) for h in hits
        ]
        trace.returned_citations = len(citations)

        cache_payload = {
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
        }
        stage_started = perf_counter()
        self.cache.put_exact(cache_key, cache_payload)
        self.cache.put_semantic(request.mode, params_key, qvec, cache_payload)
        trace.add_stage_timing("cache_write", (perf_counter() - stage_started) * 1000)
        return answer, citations, trace
