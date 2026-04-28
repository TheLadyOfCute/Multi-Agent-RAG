from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.server.state import RuntimeState
from src.cache.redis_cache import RedisCacheService
from src.use_cases.chat import RunChatQueryUseCase
from src.use_cases.documents import ClearAllDataUseCase


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    def incr(self, key: str) -> int:
        value = int(self.store.get(key, "0")) + 1
        self.store[key] = str(value)
        return value

    def scan_iter(self, match: str):
        prefix = match[:-1] if match.endswith("*") else match
        for key in list(self.store):
            if key.startswith(prefix):
                yield key

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                deleted += 1
        return deleted


def test_redis_cache_service_tracks_hits_misses_and_clear() -> None:
    service = RedisCacheService(client=FakeRedis(), enabled=True, ttl=60)

    assert service.get_embedding("doc-1") is None

    service.set_embedding("doc-1", [0.1, 0.2])
    service.set_query_embedding("query-1", [0.3, 0.4])
    service.set_answer("hello", "state-1", {"message": {"content": "cached"}})

    assert service.get_embedding("doc-1") == [0.1, 0.2]
    assert service.get_query_embedding("query-1") == [0.3, 0.4]
    assert service.get_answer("hello", "state-1") == {"message": {"content": "cached"}}

    stats = service.get_stats()
    assert stats["cache_hits"] == 3
    assert stats["cache_misses"] == 1
    assert stats["cache_size"] == 3

    assert service.clear_answer_cache() == 1
    assert service.get_answer("hello", "state-1") is None


def test_chat_query_uses_answer_cache_without_opening_stores(monkeypatch) -> None:
    runtime_state = RuntimeState()
    runtime_state.rag_initialized = True
    runtime_state.documents = [{"name": "sample.txt", "chunks": 2, "uploaded_at": "2026-04-27 10:00:00"}]

    cache_service = RedisCacheService(client=FakeRedis(), enabled=True, ttl=60)
    state_token = cache_service.build_knowledge_state_token(runtime_state.documents)
    cache_service.set_answer(
        "hello",
        state_token,
        {
            "message": {"role": "assistant", "content": "cached answer", "citations": [], "workflow_metadata": {}},
            "chunks_retrieved": 4,
            "strategy": "cached",
            "iterations": 0,
        },
    )

    monkeypatch.setattr("src.use_cases.chat.open_vector_store", lambda: (_ for _ in ()).throw(AssertionError("should not open vector store")))
    monkeypatch.setattr("src.use_cases.chat.open_neo4j_store", lambda: (_ for _ in ()).throw(AssertionError("should not open graph store")))

    use_case = RunChatQueryUseCase(runtime_state, task_registry=object(), embedder=object(), cache_service=cache_service)
    result = use_case.process_query("hello")

    assert result["content"] == "cached answer"
    assert runtime_state.performance_tracker.get_stats()["total_queries"] == 1
    assert runtime_state.messages[-1]["content"] == "cached answer"


def test_clear_all_data_use_case_clears_answer_cache(monkeypatch) -> None:
    runtime_state = RuntimeState()

    class DummyVectorStore:
        def clear_all(self) -> None:
            return None

    class DummyTaskRegistry:
        def reset(self) -> None:
            return None

    class RecordingCache:
        def __init__(self) -> None:
            self.calls = 0

        def clear_answer_cache(self) -> int:
            self.calls += 1
            return 0

    cache_service = RecordingCache()

    monkeypatch.setattr("src.use_cases.documents.open_vector_store", lambda: DummyVectorStore())
    monkeypatch.setattr("src.use_cases.documents.close_vector_store", lambda vector_store: None)

    use_case = ClearAllDataUseCase(runtime_state, DummyTaskRegistry(), cache_service)
    result = use_case.execute()

    assert result == {"cleared": True}
    assert cache_service.calls == 1
