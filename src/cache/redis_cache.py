"""Redis-backed cache helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.config import get_settings
from src.utils.logger import setup_logger


class RedisCacheService:
    """Centralized Redis cache for embeddings, answers, and cache stats."""

    EMBEDDING_PREFIX = "cache:embedding:"
    QUERY_EMBEDDING_PREFIX = "cache:query_embedding:"
    ANSWER_PREFIX = "cache:answer:"
    HIT_KEY = "cache:stats:hits"
    MISS_KEY = "cache:stats:misses"

    def __init__(
        self,
        *,
        client: Any | None = None,
        redis_url: str | None = None,
        enabled: bool | None = None,
        ttl: int | None = None,
    ) -> None:
        self.logger = setup_logger("redis_cache")
        settings = get_settings()

        self.enabled = settings.cache_enabled if enabled is None else enabled
        self.ttl = settings.cache_ttl if ttl is None else ttl
        self.redis_url = redis_url or settings.redis_url
        self.client = None
        self.available = False

        if not self.enabled:
            self.logger.info("Redis cache disabled via configuration")
            return

        if client is not None:
            self.client = client
            self.available = True
            return

        try:
            from redis import Redis
        except ModuleNotFoundError as exc:
            self.logger.warning("Redis cache unavailable: redis package is not installed (%s)", exc)
            return

        try:
            self.client = Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self.client.ping()
            self.available = True
            self.logger.info("Redis cache connected: %s", self.redis_url)
        except Exception as exc:
            self.logger.warning("Redis cache unavailable at %s: %s", self.redis_url, exc)
            self.client = None

    def get_embedding(self, text: str) -> list[float] | None:
        return self._get_vector(self.EMBEDDING_PREFIX, text)

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        self._set_vector(self.EMBEDDING_PREFIX, text, embedding)

    def get_query_embedding(self, query: str) -> list[float] | None:
        return self._get_vector(self.QUERY_EMBEDDING_PREFIX, query)

    def set_query_embedding(self, query: str, embedding: list[float]) -> None:
        self._set_vector(self.QUERY_EMBEDDING_PREFIX, query, embedding)

    def get_answer(self, query: str, knowledge_state_token: str) -> dict[str, Any] | None:
        if not self.available:
            return None

        payload = self._safe_get(self._answer_key(query, knowledge_state_token))
        if payload is None:
            self._record_miss()
            return None

        try:
            value = json.loads(payload)
            self._record_hit()
            return value
        except json.JSONDecodeError:
            self.logger.warning("Invalid cached answer payload; treating as miss")
            self._record_miss()
            return None

    def set_answer(self, query: str, knowledge_state_token: str, payload: dict[str, Any]) -> None:
        if not self.available:
            return
        self._safe_set(self._answer_key(query, knowledge_state_token), json.dumps(payload, ensure_ascii=False))

    def clear_answer_cache(self) -> int:
        return self._delete_by_prefix(self.ANSWER_PREFIX)

    def get_stats(self) -> dict[str, Any]:
        if not self.available:
            return {
                "cache_hit_rate": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_size": 0,
            }

        hits = self._safe_int(self._safe_get(self.HIT_KEY))
        misses = self._safe_int(self._safe_get(self.MISS_KEY))
        total = hits + misses
        cache_size = (
            self._count_by_prefix(self.EMBEDDING_PREFIX)
            + self._count_by_prefix(self.QUERY_EMBEDDING_PREFIX)
            + self._count_by_prefix(self.ANSWER_PREFIX)
        )
        return {
            "cache_hit_rate": hits / total if total else 0.0,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_size": cache_size,
        }

    @staticmethod
    def build_knowledge_state_token(documents: list[dict[str, Any]]) -> str:
        if not documents:
            return "empty"

        normalized = [
            {
                "name": doc.get("name", ""),
                "chunks": doc.get("chunks", 0),
                "uploaded_at": doc.get("uploaded_at", ""),
            }
            for doc in sorted(documents, key=lambda item: str(item.get("name", "")))
        ]
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_query(query: str) -> str:
        return " ".join(query.split())

    def _get_vector(self, prefix: str, text: str) -> list[float] | None:
        if not self.available:
            return None

        payload = self._safe_get(self._vector_key(prefix, text))
        if payload is None:
            self._record_miss()
            return None

        try:
            value = json.loads(payload)
            self._record_hit()
            return value
        except json.JSONDecodeError:
            self.logger.warning("Invalid cached vector payload; treating as miss")
            self._record_miss()
            return None

    def _set_vector(self, prefix: str, text: str, embedding: list[float]) -> None:
        if not self.available:
            return
        self._safe_set(self._vector_key(prefix, text), json.dumps(embedding))

    def _vector_key(self, prefix: str, text: str) -> str:
        return prefix + self._hash_text(text)

    def _answer_key(self, query: str, knowledge_state_token: str) -> str:
        normalized = self.normalize_query(query)
        return self.ANSWER_PREFIX + knowledge_state_token + ":" + self._hash_text(normalized)

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _record_hit(self) -> None:
        self._safe_incr(self.HIT_KEY)

    def _record_miss(self) -> None:
        self._safe_incr(self.MISS_KEY)

    def _delete_by_prefix(self, prefix: str) -> int:
        if not self.available:
            return 0
        keys = list(self.client.scan_iter(match=prefix + "*"))
        if not keys:
            return 0
        return int(self.client.delete(*keys))

    def _count_by_prefix(self, prefix: str) -> int:
        if not self.available:
            return 0
        return sum(1 for _ in self.client.scan_iter(match=prefix + "*"))

    def _safe_get(self, key: str) -> str | None:
        try:
            return self.client.get(key)
        except Exception as exc:
            self.logger.warning("Redis get failed for %s: %s", key, exc)
            return None

    def _safe_set(self, key: str, value: str) -> None:
        try:
            self.client.setex(key, self.ttl, value)
        except Exception as exc:
            self.logger.warning("Redis set failed for %s: %s", key, exc)

    def _safe_incr(self, key: str) -> None:
        if not self.available:
            return
        try:
            self.client.incr(key)
        except Exception as exc:
            self.logger.warning("Redis incr failed for %s: %s", key, exc)

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
