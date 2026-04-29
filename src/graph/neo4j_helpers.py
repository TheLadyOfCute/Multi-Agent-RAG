"""Neo4j helpers shared by API use cases."""

from __future__ import annotations

from typing import Any

from src.config import get_settings


# 打开 Neo4j 图数据库存储连接
def open_neo4j_store():
    # 延迟导入 Neo4j 存储类，避免应用启动时强依赖 Neo4j 或 spaCy 模块
    from src.graph.neo4j_graph_store import Neo4jGraphStore

    # 读取 Neo4j 连接配置
    settings = get_settings()

    # 创建并返回 Neo4j 存储实例
    return Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )


# 安全关闭 Neo4j 存储连接
def close_neo4j_store(store: Any) -> None:
    # 仅在对象存在且支持 close 方法时关闭连接
    if store is not None and hasattr(store, "close"):
        try:
            # 释放 Neo4j 连接资源
            store.close()
        except Exception as exc:
            # 关闭失败时仅打印错误，不影响主流程
            print(f"Neo4j close failed: {exc}", flush=True)


# 重置运行时中的 Neo4j 状态统计
def reset_neo4j_stats(runtime_state) -> None:
    # 加锁更新共享运行时状态
    with runtime_state.lock:
        # 标记 Neo4j 当前不可用
        runtime_state.neo4j_available = False

        # 清空图节点和边数量
        runtime_state.neo4j_graph_counts = {"nodes": 0, "edges": 0}

        # 清空高频实体列表
        runtime_state.neo4j_top_entities = []

        # 清空 Neo4j 错误信息
        runtime_state.neo4j_error = ""


# 查询 Neo4j 统计信息并写入运行时状态
def get_neo4j_stats(runtime_state) -> dict[str, Any]:
    # 直接查询 Neo4j 当前图谱统计
    data = _query_neo4j_stats_direct()

    # 加锁同步统计结果到运行时状态
    with runtime_state.lock:
        # 更新 Neo4j 可用状态
        runtime_state.neo4j_available = data.get("available", False)

        # 更新节点和边数量
        runtime_state.neo4j_graph_counts = data.get("counts", {"nodes": 0, "edges": 0})

        # 更新高频实体列表，并转换为 tuple 结构
        runtime_state.neo4j_top_entities = [tuple(e) for e in data.get("top_entities", [])]

        # 查询成功后清空错误信息
        runtime_state.neo4j_error = ""

    # 返回原始统计数据
    return data


# 尽力刷新 Neo4j 统计信息，失败时不向外抛出异常
def refresh_neo4j_stats_best_effort(runtime_state) -> None:
    try:
        # 尝试正常刷新 Neo4j 统计信息
        get_neo4j_stats(runtime_state)
    except Exception as exc:
        # 查询失败时先重置 Neo4j 状态
        reset_neo4j_stats(runtime_state)

        # 记录失败原因到运行时状态
        with runtime_state.lock:
            runtime_state.neo4j_error = str(exc)


# 直接连接 Neo4j 并查询图谱统计信息
def _query_neo4j_stats_direct() -> dict[str, Any]:
    # 延迟导入 Neo4j 官方驱动
    from neo4j import GraphDatabase

    # 读取 Neo4j 连接配置
    settings = get_settings()

    # 创建 Neo4j 驱动连接
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

    try:
        # 创建会话并执行统计查询
        with driver.session() as session:
            # 查询 Entity 节点数量和 RELATED_TO 关系数量
            count_row = session.run(
                "MATCH (e:Entity) "
                "WITH count(e) AS nodes "
                "OPTIONAL MATCH ()-[r:RELATED_TO]->() "
                "RETURN nodes, count(r) AS edges"
            ).single()

            # 如果没有节点，则认为图谱当前不可用或为空
            if not count_row or int(count_row["nodes"]) == 0:
                return {"available": False, "counts": {"nodes": 0, "edges": 0}, "top_entities": []}

            # 查询连接度最高的前 5 个实体
            top_rows = session.run(
                "MATCH (e:Entity)-[r:RELATED_TO]-() "
                "RETURN e.name AS name, count(r) AS degree "
                "ORDER BY degree DESC "
                "LIMIT 5"
            )

            # 组装 Neo4j 图谱统计结果
            return {
                "available": True,
                "counts": {"nodes": int(count_row["nodes"]), "edges": int(count_row["edges"])},
                "top_entities": [[row["name"], float(row["degree"])] for row in top_rows],
            }
    finally:
        # 无论查询是否成功，都关闭驱动连接
        driver.close()