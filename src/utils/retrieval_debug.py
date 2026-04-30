"""Shared formatting helpers for retrieval debug output."""

from typing import Any, Mapping, Optional


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    #dict、defaultdict、OrderedDict 都属于 Mapping（映射类型）
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(value: Any) -> Mapping[str, Any]:
    meta = _get_attr_or_key(value, "metadata", {}) or {}
    return meta if isinstance(meta, Mapping) else {}


def _short(value: Any, max_len: int = 40) -> str:
    text = str(value) if value not in (None, "") else "-"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def chunk_dedup_key(value: Any) -> str:
    """返回 chunk 的去重标识，所有入口均保证 chunk_id 存在。"""
    return _get_attr_or_key(value, "chunk_id", "")


def format_ranked_chunk_line(
    rank: int,
    value: Any,
    *,
    score: Optional[float] = None,
    include_query: bool = False,
) -> str:
    """Format one ranked retrieval result for backend logs."""
    meta = _metadata(value)
    chunk_id = _get_attr_or_key(value, "chunk_id", "")
    resolved_score = score
    if resolved_score is None:
        resolved_score = _get_attr_or_key(value, "score", None)

    score_text = "n/a" if resolved_score is None else f"{float(resolved_score):.4f}"
    retriever = meta.get("retriever") or meta.get("source")

    parts = [f"   chunk={_short(chunk_id)}", f"score={score_text}"]
    if retriever:
        parts.append(f"retriever={retriever}")
    if meta.get("sub_query_idx") is not None:
        parts.append(f"sub_query_idx={meta.get('sub_query_idx')}")
    if include_query and meta.get("query_used"):
        parts.append(f"query={_short(meta.get('query_used'), 64)}")

    return " | ".join(parts)
