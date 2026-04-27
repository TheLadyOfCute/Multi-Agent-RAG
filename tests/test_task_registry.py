from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_task_registry_tracks_success() -> None:
    from src.app.dependencies import get_runtime_state, get_task_registry

    state = get_runtime_state()
    registry = get_task_registry()
    state.reset_runtime()
    registry.reset()

    def runner(task_id, update_task):
        update_task(task_id, progress=0.5, current=1, total=2, stage="half")
        return {"ok": True}

    task_id = registry.submit_task("unit", runner)
    deadline = time.time() + 5
    task = registry.get_task(task_id)
    while task["status"] not in {"done", "error"} and time.time() < deadline:
        time.sleep(0.05)
        task = registry.get_task(task_id)

    assert task["status"] == "done"
    assert task["progress"] == 1.0
    assert task["result"] == {"ok": True}


def test_task_registry_tracks_errors() -> None:
    from src.app.dependencies import get_runtime_state, get_task_registry

    state = get_runtime_state()
    registry = get_task_registry()
    state.reset_runtime()
    registry.reset()

    def runner(task_id, update_task):
        raise RuntimeError("boom")

    task_id = registry.submit_task("unit", runner)
    deadline = time.time() + 5
    task = registry.get_task(task_id)
    while task["status"] not in {"done", "error"} and time.time() < deadline:
        time.sleep(0.05)
        task = registry.get_task(task_id)

    assert task["status"] == "error"
    assert task["error"] == "boom"
    assert "RuntimeError" in task["traceback"]
