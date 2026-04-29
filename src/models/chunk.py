"""Unified single-chunk data model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str = "unknown"
    tokens: List[int] = field(default_factory=list)
    token_count: int = 0
    start_idx: int = 0
    end_idx: int = 0
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tokens and self.token_count == 0:
            self.token_count = len(self.tokens)
        if self.start_char == 0 and self.start_idx > 0:
            self.start_char = self.start_idx
        if self.end_char == 0 and self.end_idx > 0:
            self.end_char = self.end_idx
        if self.start_idx == 0 and self.start_char > 0:
            self.start_idx = self.start_char
        if self.end_idx == 0 and self.end_char > 0:
            self.end_idx = self.end_char

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, tokens={self.token_count}, score={self.score:.3f})"

    def __str__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"[CHUNK] {preview}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        clean = dict(data)
        clean.pop("chunk_type", None)
        clean.pop("parent_id", None)
        clean.pop("children_ids", None)
        return cls(**clean)
