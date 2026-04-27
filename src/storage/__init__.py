"""Storage module for vector and metadata persistence."""

from __future__ import annotations

from .database import DatabaseManager, get_db_manager


def __getattr__(name: str):
    if name == "ChromaVectorStore":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(name)


__all__ = ["ChromaVectorStore", "DatabaseManager", "get_db_manager"]
