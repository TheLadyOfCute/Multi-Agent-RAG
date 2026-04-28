"""Background task registry abstraction."""

from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable

"""
    进程内后台任务调度与状态管理模块
"""

"""
    定义了任务函数的格式(str, Callable[..., None]) -> Any
    第一个参数是 str
    第二个参数是一个可调用对象 Callable[..., None]
    ... 表示这个回调函数可以接受任意参数
    None 表示它没有返回值
    返回值可以是任意类型 Any
"""
TaskCallable = Callable[[str, Callable[..., None]], Any]


class TaskRegistry:
    def __init__(self) -> None:
        # 存储所有后台任务的状态信息
        self._tasks: dict[str, dict[str, Any]] = {}

        # 创建线程池，用于异步执行后台任务
        self._executor = ThreadPoolExecutor(max_workers=2)

    def submit_task(self, task_type: str, runner: TaskCallable) -> str:
        # 生成唯一任务 ID
        task_id = str(uuid.uuid4())

        # 记录任务提交日志
        self._log_task(task_id, task_type, "submitted")

        # 初始化任务状态快照
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

        # 将任务提交到线程池中异步执行
        # submit(fn, *args, **kwargs)第一个为函数，后面为函数参数
        self._executor.submit(self._run_task, task_id, runner)

        # 返回任务 ID，供前端轮询状态
        return task_id

    def update_task(self, task_id: str, **updates: Any) -> None:
        # 如果任务存在，则更新任务状态
        if task_id in self._tasks:
            self._tasks[task_id].update(updates)

            # 获取任务类型用于日志记录
            task_type = self._tasks[task_id].get("type", "unknown")
        else:
            # 任务不存在时使用默认类型
            task_type = "unknown"

        # 关键字段发生变化时记录进度日志
        if {"stage", "progress", "current", "total", "last_id", "status"} & set(updates):
            self._log_task(task_id, str(task_type), "progress", **updates)

    def get_task(self, task_id: str) -> dict[str, Any]:
        # 返回任务状态副本，避免外部直接修改内部状态
        return dict(self._tasks.get(task_id, {"task_id": task_id, "status": "not_found"}))

    def task_running(self, runtime_state) -> bool:
        # 加锁读取运行时状态，判断是否有任务正在运行
        with runtime_state.lock:
            return bool(
                runtime_state.processing
                or runtime_state.ragas_evaluation_running
                or runtime_state.eval_task_id
            )

    def reset(self) -> None:
        # 清空所有任务状态记录
        self._tasks = {}

    def _run_task(self, task_id: str, runner: TaskCallable) -> None:
        # 标记任务开始运行
        self.update_task(task_id, status="running", started_at=datetime.now().isoformat())

        try:
            # 执行实际任务逻辑，并传入任务 ID 和状态更新函数
            result = runner(task_id, self.update_task)
        except Exception as exc:
            # 任务失败时记录错误、堆栈和完成时间
            self.update_task(
                task_id,
                status="error",
                error=str(exc),
                traceback=traceback.format_exc(),
                finished_at=datetime.now().isoformat(),
            )

            # 获取任务类型用于失败日志
            task_type = self._tasks.get(task_id, {}).get("type", "unknown")

            # 记录任务失败日志
            self._log_task(task_id, str(task_type), "failed", error=str(exc))
        else:
            # 任务成功时记录结果并标记完成
            self.update_task(
                task_id,
                status="done",
                result=result,
                progress=1.0,
                finished_at=datetime.now().isoformat(),
            )

            # 获取任务类型用于完成日志
            task_type = self._tasks.get(task_id, {}).get("type", "unknown")

            # 记录任务完成日志
            self._log_task(task_id, str(task_type), "completed")

    @staticmethod
    def _log_task(task_id: str, task_type: str, event: str, **fields: Any) -> None:
        # 生成当前日志时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建基础日志字段
        parts = [f"{timestamp} [task]", f"type={task_type}", f"id={task_id}", f"event={event}"]

        # 按固定顺序追加可选日志字段
        for key in ("status", "stage", "progress", "current", "total", "last_id", "error"):
            # 获取当前字段值
            value = fields.get(key)

            # 跳过空值字段
            if value is None or value == "":
                continue

            # 将进度值转换为百分比格式
            if key == "progress":
                try:
                    value = f"{float(value) * 100:.0f}%"
                except (TypeError, ValueError):
                    # 转换失败时保留原始值
                    pass

            # 追加格式化后的日志字段
            parts.append(f"{key}={value}")

        # 输出完整任务日志并立即刷新
        print(" | ".join(parts), flush=True)