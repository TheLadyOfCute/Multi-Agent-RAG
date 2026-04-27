"""Process-local runtime state."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from src.app.metrics import PerformanceTracker
"""
定义运行时的全局状态
"""

@dataclass
class SessionState:
    # 存储当前会话中的文档数据
    documents: list[dict[str, Any]] = field(default_factory=list)

    # 存储当前会话中的消息记录
    messages: list[dict[str, Any]] = field(default_factory=list)

    # 标记 RAG 是否已初始化
    rag_initialized: bool = False

    # 标记当前是否正在处理任务
    processing: bool = False


@dataclass
class SystemStatus:
    # 存储状态恢复结果说明
    restore_status: str = ""

    # 标记 Neo4j 是否可用
    neo4j_available: bool = False

    # 存储 Neo4j 图数据库节点和边的数量
    neo4j_graph_counts: dict[str, int] = field(default_factory=lambda: {"nodes": 0, "edges": 0})

    # 存储 Neo4j 中的高频实体或核心实体
    neo4j_top_entities: list[Any] = field(default_factory=list)

    # 存储 Neo4j 连接或查询错误信息
    neo4j_error: str = ""

    # 存储当前评估任务 ID
    eval_task_id: str | None = None

    # 标记 RAGAS 评估是否正在运行
    ragas_evaluation_running: bool = False


@dataclass
class MetricsStore:
    # 存储性能追踪器实例
    performance_tracker: PerformanceTracker = field(default_factory=PerformanceTracker)


class RuntimeState:
    def __init__(self) -> None:
        # 创建可重入锁，用于保护运行时状态的并发访问
        self.lock = threading.RLock()

        # 初始化会话状态
        self.session = SessionState()

        # 初始化系统状态
        self.system = SystemStatus()

        # 初始化指标存储
        self.metrics = MetricsStore()

    def reset_runtime(self) -> None:
        # 加锁重置全部运行时状态
        with self.lock:
            # 重建会话状态
            self.session = SessionState()

            # 重建系统状态
            self.system = SystemStatus()

            # 重建指标存储
            self.metrics = MetricsStore()

    def snapshot_messages(self) -> list[dict[str, Any]]:
        # 加锁复制当前消息列表，避免外部修改原始状态
        with self.lock:
            return [dict(message) for message in self.session.messages]

    def clear_messages(self) -> None:
        # 加锁清空当前会话消息
        with self.lock:
            self.session.messages = []

    @property
    def documents(self):
        # 返回当前会话文档列表
        return self.session.documents

    @documents.setter
    def documents(self, value):
        # 设置当前会话文档列表
        self.session.documents = value

    @property
    def messages(self):
        # 返回当前会话消息列表
        return self.session.messages

    @messages.setter
    def messages(self, value):
        # 设置当前会话消息列表
        self.session.messages = value

    @property
    def rag_initialized(self):
        # 返回 RAG 初始化状态
        return self.session.rag_initialized

    @rag_initialized.setter
    def rag_initialized(self, value):
        # 设置 RAG 初始化状态
        self.session.rag_initialized = value

    @property
    def processing(self):
        # 返回任务处理状态
        return self.session.processing

    @processing.setter
    def processing(self, value):
        # 设置任务处理状态
        self.session.processing = value

    @property
    def restore_status(self):
        # 返回状态恢复结果
        return self.system.restore_status

    @restore_status.setter
    def restore_status(self, value):
        # 设置状态恢复结果
        self.system.restore_status = value

    @property
    def neo4j_available(self):
        # 返回 Neo4j 可用状态
        return self.system.neo4j_available

    @neo4j_available.setter
    def neo4j_available(self, value):
        # 设置 Neo4j 可用状态
        self.system.neo4j_available = value

    @property
    def neo4j_graph_counts(self):
        # 返回 Neo4j 图数据统计
        return self.system.neo4j_graph_counts

    @neo4j_graph_counts.setter
    def neo4j_graph_counts(self, value):
        # 设置 Neo4j 图数据统计
        self.system.neo4j_graph_counts = value

    @property
    def neo4j_top_entities(self):
        # 返回 Neo4j 核心实体列表
        return self.system.neo4j_top_entities

    @neo4j_top_entities.setter
    def neo4j_top_entities(self, value):
        # 设置 Neo4j 核心实体列表
        self.system.neo4j_top_entities = value

    @property
    def neo4j_error(self):
        # 返回 Neo4j 错误信息
        return self.system.neo4j_error

    @neo4j_error.setter
    def neo4j_error(self, value):
        # 设置 Neo4j 错误信息
        self.system.neo4j_error = value

    @property
    def eval_task_id(self):
        # 返回当前评估任务 ID
        return self.system.eval_task_id

    @eval_task_id.setter
    def eval_task_id(self, value):
        # 设置当前评估任务 ID
        self.system.eval_task_id = value

    @property
    def ragas_evaluation_running(self):
        # 返回 RAGAS 评估运行状态
        return self.system.ragas_evaluation_running

    @ragas_evaluation_running.setter
    def ragas_evaluation_running(self, value):
        # 设置 RAGAS 评估运行状态
        self.system.ragas_evaluation_running = value

    @property
    def performance_tracker(self):
        # 返回性能追踪器实例
        return self.metrics.performance_tracker

    @performance_tracker.setter
    def performance_tracker(self, value):
        # 设置性能追踪器实例
        self.metrics.performance_tracker = value
