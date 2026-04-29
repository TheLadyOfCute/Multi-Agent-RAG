"""Storage module for vector and metadata persistence."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "ChromaVectorStore":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(name)


__all__ = ["ChromaVectorStore"]
