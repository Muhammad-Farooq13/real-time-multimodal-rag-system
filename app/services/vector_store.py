from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import settings


@dataclass
class SearchResult:
    doc_id: str
    score: float
    text: str


class VectorStore(Protocol):
    def add(self, vector: np.ndarray, payload: dict) -> None:
        ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vectors: list[np.ndarray] = []
        self._payloads: list[dict] = []

    def add(self, vector: np.ndarray, payload: dict) -> None:
        self._vectors.append(vector)
        self._payloads.append(payload)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if not self._vectors:
            return []

        matrix = np.vstack(self._vectors)
        # vectors are normalized in the embedding layer, so cosine equals dot product.
        scores = matrix @ query_vector
        idx = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for i in idx:
            payload = self._payloads[int(i)]
            results.append(
                SearchResult(
                    doc_id=str(payload["doc_id"]),
                    score=float(scores[int(i)]),
                    text=str(payload["text"]),
                )
            )
        return results


class FaissVectorStore:
    def __init__(self) -> None:
        self._faiss = None
        self._index = None
        self._payloads: list[dict] = []
        self._dimension: int | None = None
        self._load_faiss()

    def _load_faiss(self) -> None:
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except Exception:
            self._faiss = None

    def _ensure_index(self, vector: np.ndarray) -> None:
        if self._faiss is None:
            return
        if self._index is not None:
            return
        self._dimension = int(vector.shape[0])
        self._index = self._faiss.IndexFlatIP(self._dimension)

    def add(self, vector: np.ndarray, payload: dict) -> None:
        if self._faiss is None:
            return
        vec = np.asarray(vector, dtype=np.float32)
        self._ensure_index(vec)
        if self._index is None:
            return
        self._index.add(vec.reshape(1, -1))
        self._payloads.append(payload)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._faiss is None or self._index is None or not self._payloads:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, ids = self._index.search(q, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self._payloads):
                continue
            payload = self._payloads[int(idx)]
            results.append(
                SearchResult(
                    doc_id=str(payload["doc_id"]),
                    score=float(score),
                    text=str(payload["text"]),
                )
            )
        return results


class PineconeVectorStore:
    def __init__(self) -> None:
        self._enabled = False
        self._client = None
        self._index = None
        self._namespace = settings.pinecone_namespace
        self._init_client()

    def _list_index_names(self) -> set[str]:
        if self._client is None:
            return set()
        indexes = self._client.list_indexes()
        if hasattr(indexes, "names"):
            return set(indexes.names())
        names: set[str] = set()
        try:
            for item in indexes:
                if isinstance(item, dict) and "name" in item:
                    names.add(str(item["name"]))
        except TypeError:
            return set()
        return names

    def _ensure_index(self, dimension: int) -> None:
        if self._client is None:
            return
        from pinecone import ServerlessSpec  # type: ignore

        existing = self._list_index_names()
        if settings.pinecone_index_name not in existing:
            self._client.create_index(
                name=settings.pinecone_index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self._index = self._client.Index(settings.pinecone_index_name)

    def _init_client(self) -> None:
        if not settings.pinecone_api_key:
            return
        try:
            from pinecone import Pinecone  # type: ignore

            self._client = Pinecone(api_key=settings.pinecone_api_key)
            self._enabled = True
        except Exception:
            self._enabled = False
            self._client = None
            self._index = None

    def add(self, vector: np.ndarray, payload: dict) -> None:
        if not self._enabled:
            return
        if self._index is None:
            self._ensure_index(int(np.asarray(vector, dtype=np.float32).shape[0]))
        if self._index is None:
            return
        doc_id = str(payload["doc_id"])
        metadata = {
            "text": str(payload["text"]),
            "modality": str(payload.get("modality", "text")),
        }
        self._index.upsert(
            vectors=[
                {
                    "id": doc_id,
                    "values": np.asarray(vector, dtype=np.float32).tolist(),
                    "metadata": metadata,
                }
            ],
            namespace=self._namespace,
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if not self._enabled or self._index is None:
            return []
        response = self._index.query(
            vector=np.asarray(query_vector, dtype=np.float32).tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
        )
        matches = getattr(response, "matches", []) or []

        results: list[SearchResult] = []
        for item in matches:
            metadata = getattr(item, "metadata", {}) or {}
            results.append(
                SearchResult(
                    doc_id=str(getattr(item, "id", "")),
                    score=float(getattr(item, "score", 0.0)),
                    text=str(metadata.get("text", "")),
                )
            )
        return results


def create_vector_store() -> VectorStore:
    backend = settings.vector_backend.lower().strip()
    if backend == "pinecone":
        store = PineconeVectorStore()
        if getattr(store, "_enabled", False):
            return store
        return InMemoryVectorStore()
    if backend == "faiss":
        store = FaissVectorStore()
        if getattr(store, "_faiss", None) is not None:
            return store
        return InMemoryVectorStore()
    return InMemoryVectorStore()
