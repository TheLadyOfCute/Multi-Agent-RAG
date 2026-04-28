"""Chat-related application use cases."""

from __future__ import annotations

import time
import logging
from typing import Any, Callable

from src.server.utils.state import RuntimeState
from src.cache.redis_cache import RedisCacheService
from src.graph.neo4j_helpers import close_neo4j_store, open_neo4j_store
from src.storage.factory import close_vector_store, open_vector_store
from src.utils.logger import setup_logger
from src.utils.persistence_restore import restore_or_rebuild_bm25
logger = setup_logger("chat_use_cases") or logging.getLogger("chat_use_cases")


class GetMessagesUseCase:
    def __init__(self, runtime_state: RuntimeState):
        self.runtime_state = runtime_state

    def execute(self) -> list[dict[str, Any]]:
        return self.runtime_state.snapshot_messages()


class ClearMessagesUseCase:
    def __init__(self, runtime_state: RuntimeState):
        self.runtime_state = runtime_state

    def execute(self) -> dict[str, bool]:
        self.runtime_state.clear_messages()
        return {"cleared": True}


class ExportChatHistoryUseCase:
    def __init__(self, runtime_state: RuntimeState):
        self.runtime_state = runtime_state

    def execute(self) -> str:
        lines: list[str] = []
        for message in self.runtime_state.snapshot_messages():
            role = "User" if message.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {message.get('content', '')}")
            for citation in message.get("citations", []) or []:
                lines.append(
                    f"  Source {citation.get('source_number')}: "
                    f"{citation.get('filename', 'unknown')} "
                    f"score={citation.get('score', 0):.4f}"
                )
            lines.append("")
        return "\n".join(lines).strip() + "\n"


class RunChatQueryUseCase:
    def __init__(self, runtime_state: RuntimeState, task_registry, embedder: Any, cache_service: RedisCacheService):
        self.runtime_state = runtime_state
        self.task_registry = task_registry
        self.embedder = embedder
        self.cache_service = cache_service

    def submit_query_task(self, query: str) -> str:
        with self.runtime_state.lock:
            if not self.runtime_state.rag_initialized:
                raise RuntimeError("请先上传文档")
            self.runtime_state.messages.append({"role": "user", "content": query})

        def runner(task_id: str, update_task: Callable[..., None]) -> dict[str, Any]:
            # append_user=False 因为在提交任务时已经把用户消息添加到会话历史了，
            # 所以这里就不需要再添加一次了，避免重复。
            return self.process_query(query, append_user=False, task_id=task_id, update_task=update_task)

        return self.task_registry.submit_task("chat_query", runner)

    def process_query(
        self,
        query: str,
        *,
        append_user: bool = True,# 是否把用户问题写入会话历史
        task_id: str | None = None,
        update_task: Callable[..., None] | None = None,
    ) -> dict[str, Any]:
        """Run one question through the full RAG workflow."""

        def progress(value: float, stage: str, label: str) -> None:
            active_logger = logger or setup_logger("chat_use_cases") or logging.getLogger("chat_use_cases")
            active_logger.info(f"task_id={task_id or '-'} | stage={stage} | {label}")
            if update_task and task_id:
                update_task(task_id, progress=value, stage=stage, last_id=label)

        with self.runtime_state.lock:
            if not self.runtime_state.rag_initialized:
                raise RuntimeError("请先上传文档")
            if append_user:
                self.runtime_state.messages.append({"role": "user", "content": query})
            use_neo4j = self.runtime_state.neo4j_available
            documents_snapshot = [dict(doc) for doc in self.runtime_state.documents]

        start = time.time()
        # 获取知识库状态hash
        knowledge_state_hash = self.cache_service.build_knowledge_state_token(documents_snapshot)
        progress(0.08, "accepted", "query accepted")

        cached_payload = self.cache_service.get_answer(query, knowledge_state_hash)
        #缓存命中，直接返回答案
        if cached_payload is not None:
            assistant_message = dict(cached_payload.get("message", {}))
            # 计算缓存命中的查询耗时
            latency = time.time() - start
            with self.runtime_state.lock:
                self.runtime_state.messages.append(assistant_message)
                self.runtime_state.performance_tracker.track_query(
                    query=query,
                    latency=latency,
                    chunks_retrieved=int(cached_payload.get("chunks_retrieved", 0)),
                    strategy=str(cached_payload.get("strategy", "cached")),
                    iterations=int(cached_payload.get("iterations", 0)),
                    cache_hit=True,
                )
            progress(1.0, "done", f"cache hit in {latency:.2f}s")
            return assistant_message

        with self.runtime_state.lock:
            self.runtime_state.processing = True

        vector_store = None
        neo4j_store = None
        try:
            progress(0.18, "opening_stores", "opening vector store and graph store")
            vector_store = open_vector_store()
            neo4j_store = open_neo4j_store() if use_neo4j else None
            bm25_index = restore_or_rebuild_bm25(vector_store, index_path="data/bm25_index.pkl")
            progress(0.32, "initializing_workflow", "initializing multi-agent workflow")
            from src.workflows.factory import create_full_rag_workflow

            workflow = create_full_rag_workflow(
                vector_store=vector_store,
                embedder=self.embedder,
                bm25_index=bm25_index,
                knowledge_graph=neo4j_store,
            )

            progress(0.52, "running_workflow", "retrieving, reranking, and generating answer")
            result = workflow.run(query)
            progress(0.88, "formatting_answer", "formatting citations and metadata")
        finally:
            with self.runtime_state.lock:
                self.runtime_state.processing = False
            close_vector_store(vector_store)
            close_neo4j_store(neo4j_store)

        citations = self._format_citations(result)
        metadata = self._format_workflow_metadata(result)
        assistant_message = {
            "role": "assistant",
            "content": str(result.get("answer") or ""),
            "citations": citations,
            "workflow_metadata": metadata,
        }
        latency = time.time() - start
        result_chunks = list(result.get("chunks") or [])
        result_strategy = self._enum_value(result.get("strategy"))
        result_metadata = result.get("metadata", {}) or {}
        self.cache_service.set_answer(#???query?????
            query,
            knowledge_state_hash,
            {
                "message": assistant_message,
                "chunks_retrieved": len(result_chunks),
                "strategy": str(result_strategy),
                "iterations": int(result_metadata.get("regeneration_count", 0)),
            },
        )
        with self.runtime_state.lock:
            self.runtime_state.messages.append(assistant_message)
            self.runtime_state.performance_tracker.track_query(
                query=query,
                latency=latency,
                chunks_retrieved=len(result_chunks),
                strategy=str(result_strategy),
                iterations=result_metadata.get("regeneration_count", 0),
                cache_hit=False,
            )

        progress(1.0, "done", f"completed in {latency:.2f}s")
        return assistant_message

    @staticmethod
    def _format_citations(result: Any) -> list[dict[str, Any]]:
        citations = []
        metadata = result.get("metadata", {}) or {}
        chunks = list(result.get("chunks") or [])
        cited_ids = metadata.get("writer", {}).get("citation_ids", [])
        if cited_ids:
            citation_numbers = [citation_id for citation_id in cited_ids if 1 <= citation_id <= len(chunks)]
        else:
            citation_numbers = list(range(1, min(len(chunks), 5) + 1))

        for i in citation_numbers:
            chunk = chunks[i - 1]
            chunk_metadata = getattr(chunk, "metadata", {})
            citations.append(
                {
                    "source_number": i,
                    "filename": chunk_metadata.get("filename", "unknown"),
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "text_preview": getattr(chunk, "text", ""),
                    "score": getattr(chunk, "score", 0.0) or 0.0,
                }
            )
        return citations

    @staticmethod
    def _format_workflow_metadata(result: Any) -> dict[str, Any]:
        strategy = RunChatQueryUseCase._enum_value(result.get("strategy"))
        decision = RunChatQueryUseCase._enum_value(result.get("critic_decision"))
        metadata = result.get("metadata", {}) or {}
        return {
            "complexity": result.get("complexity"),
            "strategy": strategy,
            "selected_retrievers": result.get("selected_retrievers", []),
            "retriever_quotas": result.get("retriever_quotas", {}),
            "retrieval_rounds": result.get("retrieval_round"),
            "validation_score": result.get("validation_score"),
            "critic_score": result.get("critic_score"),
            "regenerations": metadata.get("regeneration_count", 0),
            "decision": str(decision).upper() if decision is not None else "",
            "reranker_used_cohere": metadata.get("reranker", {}).get("used_cohere", False),
        }

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return value.value if hasattr(value, "value") else value
