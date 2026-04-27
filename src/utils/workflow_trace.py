"""Readable trace formatting for the RAG workflow."""

from collections import Counter
from typing import Any, Dict, Iterable


def format_stage_trace(
    stage: str,
    inputs: Dict[str, Any] | None = None,
    outputs: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
) -> str:
    """Format a compact stage trace with input, output, and metrics."""
    parts = [f"▶ {stage.upper()}"]

    if inputs:
        parts.append(f"input: {_format_pairs(inputs)}")
    if outputs:
        parts.append(f"output: {_format_pairs(outputs)}")
    if metrics:
        parts.append(_format_pairs(metrics))

    return " | ".join(parts)


def summarize_chunks(chunks: Iterable[Any]) -> str:
    """Summarize chunk types and retriever/source counts."""
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_type = metadata.get("chunk_type") or getattr(chunk, "chunk_type", "unknown")
        source = metadata.get("retriever") or metadata.get("source") or "unknown"
        type_counts[str(chunk_type)] += 1
        source_counts[str(source)] += 1

    return f"types={dict(type_counts)}, sources={dict(source_counts)}"


def _format_pairs(values: Dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items())
