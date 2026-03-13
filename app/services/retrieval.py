from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.vector_store import SearchResult


@dataclass
class HybridRetriever:
    corpus_rows: list[dict]

    def __post_init__(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._texts = [str(row.get("text", "")) for row in self.corpus_rows]
        self._doc_ids = [str(row.get("doc_id", "")) for row in self.corpus_rows]
        self._sparse_matrix = self._vectorizer.fit_transform(self._texts) if self._texts else None

    def blend(
        self,
        query_text: str,
        dense_hits: list[SearchResult],
        top_k: int,
        alpha: float,
    ) -> list[SearchResult]:
        if not dense_hits and self._sparse_matrix is None:
            return []

        alpha = float(max(0.0, min(1.0, alpha)))
        dense_map = {h.doc_id: h for h in dense_hits}
        sparse_scores: dict[str, float] = {}

        if self._sparse_matrix is not None and query_text.strip():
            query_vec = self._vectorizer.transform([query_text])
            sim = (self._sparse_matrix @ query_vec.T).toarray().reshape(-1)
            if sim.size > 0:
                top_ids = np.argsort(sim)[::-1][: max(top_k * 3, top_k)]
                for idx in top_ids:
                    doc_id = self._doc_ids[int(idx)]
                    sparse_scores[doc_id] = float(sim[int(idx)])

        max_dense = max((h.score for h in dense_hits), default=1.0)
        max_sparse = max(sparse_scores.values(), default=1.0)
        if max_dense == 0:
            max_dense = 1.0
        if max_sparse == 0:
            max_sparse = 1.0

        all_doc_ids = set(dense_map.keys()) | set(sparse_scores.keys())
        merged: list[SearchResult] = []
        for doc_id in all_doc_ids:
            dense_score = dense_map[doc_id].score / max_dense if doc_id in dense_map else 0.0
            sparse_score = sparse_scores.get(doc_id, 0.0) / max_sparse
            score = alpha * dense_score + (1.0 - alpha) * sparse_score

            text = dense_map[doc_id].text if doc_id in dense_map else self._lookup_text(doc_id)
            merged.append(SearchResult(doc_id=doc_id, score=float(score), text=text))

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def _lookup_text(self, doc_id: str) -> str:
        for row in self.corpus_rows:
            if str(row.get("doc_id", "")) == doc_id:
                return str(row.get("text", ""))
        return ""
