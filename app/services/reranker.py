from __future__ import annotations

import numpy as np

from app.services.embeddings import Embedder
from app.services.vector_store import SearchResult


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._cross_encoder = None

    def _load_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
        except Exception:
            self._cross_encoder = None
        return self._cross_encoder

    def rerank(
        self,
        query_text: str,
        candidates: list[SearchResult],
        top_n: int,
        embedder: Embedder,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        top_n = max(1, min(top_n, len(candidates)))
        model = self._load_cross_encoder()

        if model is not None:
            pairs = [[query_text, hit.text] for hit in candidates]
            scores = model.predict(pairs)
            reranked = [
                SearchResult(doc_id=hit.doc_id, score=float(score), text=hit.text)
                for hit, score in zip(candidates, scores)
            ]
            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:top_n]

        # Fallback rerank: cosine similarity in embedding space.
        qvec = embedder.embed(query_text)
        scored: list[SearchResult] = []
        for hit in candidates:
            dvec = embedder.embed(hit.text)
            score = float(np.dot(qvec, dvec))
            scored.append(SearchResult(doc_id=hit.doc_id, score=score, text=hit.text))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]
