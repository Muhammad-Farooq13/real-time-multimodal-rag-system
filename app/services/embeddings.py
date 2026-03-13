from __future__ import annotations

import hashlib

import numpy as np

from app.core.config import settings


class Embedder:
    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
        except Exception:
            self._model = None
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._load_model()
        if model is not None:
            vec = model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vec, dtype=np.float32)

        # Deterministic fallback vector to keep local dev unblocked.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest * 16, dtype=np.uint8)[:384].astype(np.float32)
        norm = np.linalg.norm(raw)
        if norm == 0:
            return raw
        return raw / norm
