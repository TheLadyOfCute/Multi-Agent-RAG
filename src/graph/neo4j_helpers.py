"""Neo4j helpers shared by API use cases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.config import get_settings


def open_neo4j_store():
    # Lazy import keeps app startup independent from Neo4j/spaCy modules.
    from src.graph.neo4j_graph_store import Neo4jGraphStore

    settings = get_settings()
    return Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )


def close_neo4j_store(store: Any) -> None:
    if store is not None and hasattr(store, "close"):
        try:
            store.close()
        except Exception as exc:
            print(f"Neo4j close failed: {exc}", flush=True)


def reset_neo4j_stats(runtime_state) -> None:
    with runtime_state.lock:
        runtime_state.neo4j_available = False
        runtime_state.neo4j_graph_counts = {"nodes": 0, "edges": 0}
        runtime_state.neo4j_top_entities = []
        runtime_state.neo4j_error = ""


def get_neo4j_stats(runtime_state) -> dict[str, Any]:
    data = _query_neo4j_stats_direct()
    with runtime_state.lock:
        runtime_state.neo4j_available = data.get("available", False)
        runtime_state.neo4j_graph_counts = data.get("counts", {"nodes": 0, "edges": 0})
        runtime_state.neo4j_top_entities = [tuple(e) for e in data.get("top_entities", [])]
        runtime_state.neo4j_error = ""
    return data


def refresh_neo4j_stats_best_effort(runtime_state) -> None:
    try:
        get_neo4j_stats(runtime_state)
    except Exception as exc:
        reset_neo4j_stats(runtime_state)
        with runtime_state.lock:
            runtime_state.neo4j_error = str(exc)


def _query_neo4j_stats_direct() -> dict[str, Any]:
    from neo4j import GraphDatabase

    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    try:
        with driver.session() as session:
            count_row = session.run(
                "MATCH (e:Entity) "
                "WITH count(e) AS nodes "
                "OPTIONAL MATCH ()-[r:RELATED_TO]->() "
                "RETURN nodes, count(r) AS edges"
            ).single()
            if not count_row or int(count_row["nodes"]) == 0:
                return {"available": False, "counts": {"nodes": 0, "edges": 0}, "top_entities": []}

            top_rows = session.run(
                "MATCH (e:Entity)-[r:RELATED_TO]-() "
                "RETURN e.name AS name, count(r) AS degree "
                "ORDER BY degree DESC "
                "LIMIT 5"
            )
            return {
                "available": True,
                "counts": {"nodes": int(count_row["nodes"]), "edges": int(count_row["edges"])},
                "top_entities": [[row["name"], float(row["degree"])] for row in top_rows],
            }
    finally:
        driver.close()


def build_neo4j_graph_subprocess(chunks: list[Any]) -> None:
    # Run graph building in a child process to isolate native dependency issues.
    payload = {
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count,
                "start_idx": chunk.start_idx,
                "end_idx": chunk.end_idx,
                "chunk_type": chunk.chunk_type,
            }
            for chunk in chunks
        ]
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as file:
            json.dump(payload, file, ensure_ascii=False)
            temp_path = file.name

        result = subprocess.run(
            [sys.executable, "-m", "src.graph.build_neo4j_graph", temp_path],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=300,
        )
        if result.stdout:
            print(result.stdout.strip(), flush=True)
        if result.stderr:
            print(result.stderr.strip(), flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"Neo4j graph subprocess failed with exit code {result.returncode}")
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
