from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass

import numpy as np

from app.core.config import settings


@dataclass
class SemanticEntry:
    created_at: float
    mode: str
    params_key: str
    query_vector: np.ndarray
    payload: dict


class ResponseCache:
    def __init__(self) -> None:
        self.enabled = settings.cache_enabled
        self.ttl_seconds = max(1, settings.cache_ttl_seconds)
        self.semantic_threshold = float(settings.semantic_cache_threshold)
        self.max_entries = max(100, settings.semantic_cache_max_entries)
        self.distributed_semantic = settings.semantic_cache_distributed
        self.backend = settings.cache_backend.strip().lower()
        self._redis = None
        self._vector_index_name = settings.redis_vector_index_name
        self._vector_prefix = settings.redis_vector_prefix
        self._vector_search_k = max(1, settings.redis_vector_search_k)
        self._vector_index_ready = False
        self._vector_dim: int | None = None
        self._exact_local: dict[str, tuple[float, dict]] = {}
        self._semantic_entries: list[SemanticEntry] = []
        if self.enabled and self.backend == "redis":
            self._connect_redis()

    def _connect_redis(self) -> None:
        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
            client.ping()
            self._redis = client
        except Exception:
            self._redis = None

    @staticmethod
    def _decode(value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value

    @staticmethod
    def _escape_tag_value(value: str) -> str:
        specials = set("-{}[]()|@:\\")
        return "".join(f"\\{ch}" if ch in specials else ch for ch in value)

    def _ensure_redis_vector_index(self, dim: int) -> None:
        if self._redis is None or not self.distributed_semantic:
            return
        if self._vector_index_ready and self._vector_dim == dim:
            return

        try:
            self._redis.execute_command(
                "FT.CREATE",
                self._vector_index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._vector_prefix,
                "SCHEMA",
                "mode",
                "TAG",
                "params_key",
                "TAG",
                "created_at",
                "NUMERIC",
                "payload",
                "TEXT",
                "query_vector",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dim),
                "DISTANCE_METRIC",
                "COSINE",
            )
            self._vector_index_ready = True
            self._vector_dim = dim
        except Exception as exc:
            msg = self._decode(getattr(exc, "args", [""])[0] if getattr(exc, "args", None) else "")
            if "Index already exists" in msg:
                self._vector_index_ready = True
                self._vector_dim = dim
            else:
                self._vector_index_ready = False

    def make_params_key(self, params: dict) -> str:
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def make_cache_key(self, mode: str, params_key: str, query_text: str) -> str:
        canonical = f"{mode}|{params_key}|{query_text.strip().lower()}"
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"rag:resp:{digest}"

    def get_exact(self, key: str) -> dict | None:
        if not self.enabled:
            return None

        if self._redis is not None:
            raw = self._redis.get(key)
            if not raw:
                return None
            return json.loads(self._decode(raw))

        now = time.time()
        item = self._exact_local.get(key)
        if item is None:
            return None
        expires_at, payload = item
        if now > expires_at:
            self._exact_local.pop(key, None)
            return None
        return payload

    def put_exact(self, key: str, payload: dict) -> None:
        if not self.enabled:
            return

        if self._redis is not None:
            self._redis.setex(key, self.ttl_seconds, json.dumps(payload, ensure_ascii=True).encode("utf-8"))
            return

        self._exact_local[key] = (time.time() + self.ttl_seconds, payload)

    def get_semantic(self, mode: str, params_key: str, query_vector: np.ndarray) -> dict | None:
        if not self.enabled:
            return None

        distributed = self._get_semantic_distributed(mode, params_key, query_vector)
        if distributed is not None:
            return distributed

        self._prune_semantic_entries()
        best_score = -1.0
        best_payload: dict | None = None
        for item in self._semantic_entries:
            if item.mode != mode or item.params_key != params_key:
                continue
            score = float(np.dot(query_vector, item.query_vector))
            if score > best_score:
                best_score = score
                best_payload = item.payload

        if best_score >= self.semantic_threshold:
            return best_payload
        return None

    def _get_semantic_distributed(
        self, mode: str, params_key: str, query_vector: np.ndarray
    ) -> dict | None:
        if self._redis is None or not self.distributed_semantic:
            return None

        vec = np.asarray(query_vector, dtype=np.float32)
        self._ensure_redis_vector_index(int(vec.shape[0]))
        if not self._vector_index_ready:
            return None

        mode_tag = self._escape_tag_value(mode)
        params_tag = self._escape_tag_value(params_key)
        q = f"(@mode:{{{mode_tag}}} @params_key:{{{params_tag}}})=>[KNN {self._vector_search_k} @query_vector $vec AS score]"

        try:
            result = self._redis.execute_command(
                "FT.SEARCH",
                self._vector_index_name,
                q,
                "PARAMS",
                "2",
                "vec",
                vec.tobytes(),
                "SORTBY",
                "score",
                "RETURN",
                "2",
                "payload",
                "score",
                "DIALECT",
                "2",
            )
        except Exception:
            return None

        if not isinstance(result, list) or len(result) < 3:
            return None

        fields = result[2]
        if not isinstance(fields, list):
            return None

        field_map: dict[str, str] = {}
        for i in range(0, len(fields), 2):
            key = self._decode(fields[i])
            val = self._decode(fields[i + 1]) if i + 1 < len(fields) else ""
            field_map[key] = val

        distance = float(field_map.get("score", "1.0") or 1.0)
        similarity = max(0.0, min(1.0, 1.0 - distance))
        if similarity < self.semantic_threshold:
            return None

        payload_raw = field_map.get("payload", "")
        if not payload_raw:
            return None
        try:
            return json.loads(payload_raw)
        except json.JSONDecodeError:
            return None

    def put_semantic(self, mode: str, params_key: str, query_vector: np.ndarray, payload: dict) -> None:
        if not self.enabled:
            return

        self._put_semantic_distributed(mode, params_key, query_vector, payload)

        self._semantic_entries.append(
            SemanticEntry(
                created_at=time.time(),
                mode=mode,
                params_key=params_key,
                query_vector=np.asarray(query_vector, dtype=np.float32),
                payload=payload,
            )
        )
        if len(self._semantic_entries) > self.max_entries:
            self._semantic_entries = self._semantic_entries[-self.max_entries :]

    def _put_semantic_distributed(
        self, mode: str, params_key: str, query_vector: np.ndarray, payload: dict
    ) -> None:
        if self._redis is None or not self.distributed_semantic:
            return

        vec = np.asarray(query_vector, dtype=np.float32)
        self._ensure_redis_vector_index(int(vec.shape[0]))
        if not self._vector_index_ready:
            return

        key = f"{self._vector_prefix}{uuid.uuid4()}"
        payload_text = json.dumps(payload, ensure_ascii=True)
        try:
            self._redis.hset(
                key,
                mapping={
                    "mode": mode,
                    "params_key": params_key,
                    "created_at": str(time.time()),
                    "payload": payload_text,
                    "query_vector": vec.tobytes(),
                },
            )
            self._redis.expire(key, self.ttl_seconds)
        except Exception:
            return

    def _prune_semantic_entries(self) -> None:
        now = time.time()
        min_time = now - self.ttl_seconds
        self._semantic_entries = [entry for entry in self._semantic_entries if entry.created_at >= min_time]
