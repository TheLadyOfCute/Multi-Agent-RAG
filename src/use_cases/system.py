"""System and monitoring use cases."""

from __future__ import annotations

from src.app.state import RuntimeState
from src.cache.redis_cache import RedisCacheService
from src.graph.neo4j_helpers import get_neo4j_stats


class GetSystemStateUseCase:
    def __init__(self, runtime_state: RuntimeState, task_registry, ensure_loaded_use_case, cache_service: RedisCacheService):
        self.runtime_state = runtime_state
        self.task_registry = task_registry
        self.ensure_loaded_use_case = ensure_loaded_use_case
        self.cache_service = cache_service

    def execute(self) -> dict:
        # 这个快照是前端恢复页面和显示全局状态的主要数据来源。
        self.ensure_loaded_use_case.execute()
        with self.runtime_state.lock:
            return {
                "document_count": len(self.runtime_state.documents),
                "message_count": len(self.runtime_state.messages),
                "rag_initialized": self.runtime_state.rag_initialized,
                "restore_status": self.runtime_state.restore_status,
                "processing": self.runtime_state.processing,
                "task_running": self.task_registry.task_running(self.runtime_state),
                "eval_task_id": self.runtime_state.eval_task_id,
                "ragas_evaluation_running": self.runtime_state.ragas_evaluation_running,
                "neo4j": {
                    "available": self.runtime_state.neo4j_available,
                    "counts": self.runtime_state.neo4j_graph_counts,
                    "top_entities": self.runtime_state.neo4j_top_entities,
                    "error": self.runtime_state.neo4j_error,
                },
                "performance": _merge_performance_stats(
                    self.runtime_state.performance_tracker.get_stats(),
                    self.cache_service.get_stats(),
                ),
            }


class GetGraphStatsUseCase:
    def __init__(self, runtime_state: RuntimeState):
        self.runtime_state = runtime_state

    def execute(self) -> dict:
        return get_neo4j_stats(self.runtime_state)


class GetPerformanceStatsUseCase:
    def __init__(self, runtime_state: RuntimeState, cache_service: RedisCacheService):
        self.runtime_state = runtime_state
        self.cache_service = cache_service

    def execute(self) -> dict:
        with self.runtime_state.lock:
            return _merge_performance_stats(
                self.runtime_state.performance_tracker.get_stats(),
                self.cache_service.get_stats(),
            )


class SavePerformanceMetricsUseCase:
    def __init__(self, runtime_state: RuntimeState):
        self.runtime_state = runtime_state

    def execute(self) -> dict[str, str]:
        with self.runtime_state.lock:
            self.runtime_state.performance_tracker.save_metrics()
        return {"status": "saved"}


def _merge_performance_stats(perf_stats: dict, cache_stats: dict) -> dict:
    merged = dict(perf_stats)
    merged["cache_hit_rate"] = cache_stats.get("cache_hit_rate", 0.0)
    merged["cache_hits"] = cache_stats.get("cache_hits", 0)
    merged["cache_misses"] = cache_stats.get("cache_misses", 0)
    merged["cache_size"] = cache_stats.get("cache_size", 0)
    return merged
