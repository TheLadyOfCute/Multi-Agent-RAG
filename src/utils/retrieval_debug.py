"""Shared formatting helpers for retrieval debug output."""

from typing import Any, Mapping, Optional


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(value: Any) -> Mapping[str, Any]:
    meta = _get_attr_or_key(value, "metadata", {}) or {}
    return meta if isinstance(meta, Mapping) else {}


def _short(value: Any, max_len: int = 40) -> str:
    text = str(value) if value not in (None, "") else "-"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def chunk_identity(value: Any) -> dict[str, str]:
    """Return the visible parent/child identity for a dict result or Chunk."""
    meta = _metadata(value)
    chunk_id = _get_attr_or_key(value, "chunk_id", "")
    chunk_type = (
        meta.get("chunk_type")
        or _get_attr_or_key(value, "chunk_type", None)
        or "chunk"
    )
    child_id = (
        _get_attr_or_key(value, "child_chunk_id", None)
        or meta.get("child_chunk_id")
        or meta.get("matched_child_id")
    )
    parent_id = (
        meta.get("parent_id")
        or meta.get("parent_chunk_id")
        or _get_attr_or_key(value, "parent_id", None)
    )

    if chunk_type == "parent":
        parent_id = chunk_id or parent_id
    elif chunk_type == "child":
        child_id = child_id or chunk_id

    return {
        "chunk_type": str(chunk_type),
        "parent_id": str(parent_id or ""),
        "child_id": str(child_id or ""),
    }


def chunk_dedup_key(value: Any) -> str:
    """Return the preferred duplicate key for retrieval results."""
    identity = chunk_identity(value)
    if identity["child_id"]:
        return f"child:{identity['child_id']}"

    chunk_id = _get_attr_or_key(value, "chunk_id", "")
    if chunk_id:
        return f"chunk:{chunk_id}"

    text = _get_attr_or_key(value, "text", "") or ""
    return "text:" + " ".join(str(text).lower().split())


def merge_retriever_sources(kept: Any, duplicates: list[Any]) -> None:
    """Merge retriever/source labels from duplicate chunks into kept metadata."""
    kept_meta = _metadata(kept)
    sources = set()

    for value in [kept, *duplicates]:
        meta = _metadata(value)
        for field in ("retriever", "source"):
            raw = meta.get(field)
            if not raw:
                continue
            for item in str(raw).split("|"):
                item = item.strip()
                if item:
                    sources.add(item)

    if not sources or not isinstance(kept_meta, dict):
        return

    merged = "|".join(sorted(sources))
    kept_meta["retriever"] = merged
    kept_meta["source"] = merged


def format_ranked_chunk_line(
    rank: int,
    value: Any,
    *,
    score: Optional[float] = None,
    include_query: bool = False,
) -> str:
    """Format one ranked retrieval result for backend logs."""
    meta = _metadata(value)
    identity = chunk_identity(value)
    resolved_score = score
    if resolved_score is None:
        resolved_score = _get_attr_or_key(value, "score", None)

    score_text = "n/a" if resolved_score is None else f"{float(resolved_score):.4f}"
    retriever = meta.get("retriever") or meta.get("source")

    parts = [f"   child={_short(identity['child_id'])}", f"score={score_text}"]
    if retriever:
        parts.append(f"retriever={retriever}")
    if meta.get("sub_query_idx") is not None:
        parts.append(f"sub_query_idx={meta.get('sub_query_idx')}")
    if include_query and meta.get("query_used"):
        parts.append(f"query={_short(meta.get('query_used'), 64)}")

    return " | ".join(parts)

