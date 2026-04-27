"""Python entrypoint for running the FastAPI server with code-defined defaults.

Usage:
  python -m src.app
"""

from __future__ import annotations

import logging

import uvicorn

from src.app.state import RuntimeState
from src.config import get_settings
from src.graph.neo4j_helpers import refresh_neo4j_stats_best_effort


def main() -> None:
    settings = get_settings()
    # Uvicorn configures handlers for uvicorn.* loggers by default; use that so messages show up.
    logger = logging.getLogger("uvicorn.error")
    logger.info("Starting API (bind=%s:%s reload=%s)", settings.api_host, settings.api_port, settings.api_reload)
    temp_state = RuntimeState()
    refresh_neo4j_stats_best_effort(temp_state)
    with temp_state.lock:
        if temp_state.neo4j_available:
            counts = temp_state.neo4j_graph_counts or {"nodes": 0, "edges": 0}
            logger.info(
                "Neo4j connected (%s): nodes=%s edges=%s",
                settings.neo4j_uri,
                counts.get("nodes", 0),
                counts.get("edges", 0),
            )
        else:
            logger.warning("Neo4j unavailable (%s): %s", settings.neo4j_uri, temp_state.neo4j_error or "unknown")
    uvicorn.run(
        "src.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
