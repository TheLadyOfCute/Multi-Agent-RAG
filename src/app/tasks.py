"""Background task registry abstraction."""

from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable

TaskCallable = Callable[[str, Callable[..., None]], Any]


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def submit_task(self, task_type: str, runner: TaskCallable) -> str:
        # 任务注册器负责两件事：
        # 1. 生成 task_id 与状态快照
        # 2. 把真正执行逻辑扔进线程池
        task_id = str(uuid.uuid4())
        self._log_task(task_id, task_type, "submitted")
        self._tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "status": "pending",
            "progress": 0.0,
            "current": 0,
            "total": 0,
            "stage": "",
            "last_id": "",
            "submitted_at": datetime.now().isoformat(),
        }
        self._executor.submit(self._run_task, task_id, runner)
        return task_id

    def update_task(self, task_id: str, **updates: Any) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].update(updates)
            task_type = self._tasks[task_id].get("type", "unknown")
        else:
            task_type = "unknown"
        if {"stage", "progress", "current", "total", "last_id", "status"} & set(updates):
            self._log_task(task_id, str(task_type), "progress", **updates)

    def get_task(self, task_id: str) -> dict[str, Any]:
        return dict(self._tasks.get(task_id, {"task_id": task_id, "status": "not_found"}))

    def task_running(self, runtime_state) -> bool:
        with runtime_state.lock:
            return bool(
                runtime_state.processing
                or runtime_state.ragas_evaluation_running
                or runtime_state.eval_task_id
            )

    def reset(self) -> None:
        self._tasks = {}

    def _run_task(self, task_id: str, runner: TaskCallable) -> None:
        # 所有后台任务都经过统一的成功/失败状态收口，前端轮询逻辑会更稳定。
        self.update_task(task_id, status="running", started_at=datetime.now().isoformat())
        try:
            result = runner(task_id, self.update_task)
        except Exception as exc:
            self.update_task(
                task_id,
                status="error",
                error=str(exc),
                traceback=traceback.format_exc(),
                finished_at=datetime.now().isoformat(),
            )
            task_type = self._tasks.get(task_id, {}).get("type", "unknown")
            self._log_task(task_id, str(task_type), "failed", error=str(exc))
        else:
            self.update_task(
                task_id,
                status="done",
                result=result,
                progress=1.0,
                finished_at=datetime.now().isoformat(),
            )
            task_type = self._tasks.get(task_id, {}).get("type", "unknown")
            self._log_task(task_id, str(task_type), "completed")

    @staticmethod
    def _log_task(task_id: str, task_type: str, event: str, **fields: Any) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"{timestamp} [task]", f"type={task_type}", f"id={task_id}", f"event={event}"]
        for key in ("status", "stage", "progress", "current", "total", "last_id", "error"):
            value = fields.get(key)
            if value is None or value == "":
                continue
            if key == "progress":
                try:
                    value = f"{float(value) * 100:.0f}%"
                except (TypeError, ValueError):
                    pass
            parts.append(f"{key}={value}")
        print(" | ".join(parts), flush=True)
