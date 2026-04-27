"""FastAPI lifespan hooks."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from src.app.dependencies import get_restore_document_state_use_case, get_runtime_state
from src.config import get_settings
from src.app.runtime import configure_runtime

# 配置应用运行时环境
configure_runtime()


@asynccontextmanager
async def app_lifespan(_app):
    # 在首次请求前恢复已持久化的状态
    get_restore_document_state_use_case().restore_persisted_state()
    # 获取应用配置
    settings = get_settings()

    # 获取运行时状态
    runtime_state = get_runtime_state()

    # 获取 Uvicorn 的错误日志记录器
    logger = logging.getLogger("uvicorn.error")

    # 加锁读取运行时状态，避免并发访问导致状态不一致
    with runtime_state.lock:
        # 判断 Neo4j 当前是否可用
        if runtime_state.neo4j_available:
            # 获取 Neo4j 图数据统计，缺省时使用空统计
            counts = runtime_state.neo4j_graph_counts or {"nodes": 0, "edges": 0}

            # 记录 Neo4j 连接成功及图数据统计信息
            logger.info(
                "Neo4j connected (%s): nodes=%s edges=%s",
                settings.neo4j_uri,
                counts.get("nodes", 0),
                counts.get("edges", 0),
            )
        else:
            # 记录 Neo4j 不可用及错误原因
            logger.warning(
                "Neo4j unavailable (%s): %s",
                settings.neo4j_uri,
                runtime_state.neo4j_error or "unknown",
            )

    # 将控制权交给应用生命周期，等待应用运行结束
    yield
