"""Evaluation module for RAG system quality assessment."""

from __future__ import annotations

import importlib

from .simple_evaluator import SimpleEvaluator

__all__ = ["SimpleEvaluator", "RAGASEvaluator", "HAS_RAGAS"]


def __getattr__(name: str):
    """Load RAGAS-only dependencies lazily to keep app startup lightweight."""
    if name == "RAGASEvaluator":
        module = importlib.import_module(".ragas_evaluator", __name__)
        return module.RAGASEvaluator
    if name == "HAS_RAGAS":
        try:
            importlib.import_module(".ragas_evaluator", __name__)
        except ImportError:
            return False
        return True
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
