"""Process-local runtime state."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from src.server.utils.metrics import PerformanceTracker


@dataclass
class SessionState:
    documents: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    rag_initialized: bool = False
    processing: bool = False


@dataclass
class SystemStatus:
    restore_status: str = ""
    neo4j_available: bool = False
    neo4j_graph_counts: dict[str, int] = field(default_factory=lambda: {"nodes": 0, "edges": 0})
    neo4j_top_entities: list[Any] = field(default_factory=list)
    neo4j_error: str = ""
    eval_task_id: str | None = None
    ragas_evaluation_running: bool = False


@dataclass
class MetricsStore:
    performance_tracker: PerformanceTracker = field(default_factory=PerformanceTracker)


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.session = SessionState()
        self.system = SystemStatus()
        self.metrics = MetricsStore()
        self.persisted_restore_in_progress = False
        self.persisted_restore_done = False

    def reset_runtime(self) -> None:
        with self.lock:
            self.session = SessionState()
            self.system = SystemStatus()
            self.metrics = MetricsStore()
            self.persisted_restore_in_progress = False
            self.persisted_restore_done = False

    def snapshot_messages(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(message) for message in self.session.messages]

    def clear_messages(self) -> None:
        with self.lock:
            self.session.messages = []

    @property
    def documents(self):
        return self.session.documents

    @documents.setter
    def documents(self, value):
        self.session.documents = value

    @property
    def messages(self):
        return self.session.messages

    @messages.setter
    def messages(self, value):
        self.session.messages = value

    @property
    def rag_initialized(self):
        return self.session.rag_initialized

    @rag_initialized.setter
    def rag_initialized(self, value):
        self.session.rag_initialized = value

    @property
    def processing(self):
        return self.session.processing

    @processing.setter
    def processing(self, value):
        self.session.processing = value

    @property
    def restore_status(self):
        return self.system.restore_status

    @restore_status.setter
    def restore_status(self, value):
        self.system.restore_status = value

    @property
    def neo4j_available(self):
        return self.system.neo4j_available

    @neo4j_available.setter
    def neo4j_available(self, value):
        self.system.neo4j_available = value

    @property
    def neo4j_graph_counts(self):
        return self.system.neo4j_graph_counts

    @neo4j_graph_counts.setter
    def neo4j_graph_counts(self, value):
        self.system.neo4j_graph_counts = value

    @property
    def neo4j_top_entities(self):
        return self.system.neo4j_top_entities

    @neo4j_top_entities.setter
    def neo4j_top_entities(self, value):
        self.system.neo4j_top_entities = value

    @property
    def neo4j_error(self):
        return self.system.neo4j_error

    @neo4j_error.setter
    def neo4j_error(self, value):
        self.system.neo4j_error = value

    @property
    def eval_task_id(self):
        return self.system.eval_task_id

    @eval_task_id.setter
    def eval_task_id(self, value):
        self.system.eval_task_id = value

    @property
    def ragas_evaluation_running(self):
        return self.system.ragas_evaluation_running

    @ragas_evaluation_running.setter
    def ragas_evaluation_running(self, value):
        self.system.ragas_evaluation_running = value

    @property
    def performance_tracker(self):
        return self.metrics.performance_tracker

    @performance_tracker.setter
    def performance_tracker(self, value):
        self.metrics.performance_tracker = value
