"""System and shared endpoints (health, state, tasks, metrics)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from src.server.utils.dependencies import (
    get_graph_stats_use_case,
    get_performance_stats_use_case,
    get_runtime_state,
    get_save_performance_metrics_use_case,
    get_system_state_use_case,
    get_task_registry,
)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/state")
def get_state() -> dict[str, Any]:
    # 这是前端初始化和页面恢复时最常用的总览接口。
    return get_system_state_use_case().execute()


@router.get("/graph/stats")
def graph_stats() -> dict[str, Any]:
    logger = logging.getLogger("uvicorn.error")
    logger.info("Neo4j stats refresh requested: GET /api/graph/stats")
    try:
        result = get_graph_stats_use_case().execute()
        counts = (result or {}).get("counts") or {}
        logger.info(
            "Neo4j stats refreshed: available=%s nodes=%s edges=%s",
            bool((result or {}).get("available", False)),
            counts.get("nodes", 0),
            counts.get("edges", 0),
        )
        return result
    except Exception as exc:
        runtime_state = get_runtime_state()
        with runtime_state.lock:
            runtime_state.neo4j_error = str(exc)
        logger.warning("Neo4j stats refresh failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    return get_task_registry().get_task(task_id)


@router.get("/performance")
def performance() -> dict[str, Any]:
    return get_performance_stats_use_case().execute()


@router.post("/performance/save")
def save_performance() -> dict[str, str]:
    return get_save_performance_metrics_use_case().execute()
