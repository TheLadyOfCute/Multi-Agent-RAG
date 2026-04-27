"""
Flat chunking system.

Creates a single sequence of overlapping chunks without any parent-child
relationships. This is the true flat chunking mode used by the app.
"""

from typing import Any, Dict, List

import tiktoken

from src.models.chunk import Chunk


class FlatChunker:
    """
    Create flat chunks from text.

    Strategy:
    1. Split the full text into chunks of fixed token length
    2. Apply optional overlap between adjacent chunks
    3. Do not create parent-child relationships
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_text(
        self,
        text: str,
        doc_id: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """Split text into flat chunks."""
        print(f"\nCreating flat chunks...")
        print(f"   Chunk size: {self.chunk_size} tokens")
        print(f"   Chunk overlap: {self.chunk_overlap} tokens")

        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        print(f"   Total tokens: {total_tokens:,}")

        if total_tokens == 0:
            return []

        chunks: List[Chunk] = []
        chunk_num = 0
        start_idx = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start_idx < total_tokens:
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunk_metadata = dict(metadata or {})
            chunk_metadata["chunking_mode"] = "flat"

            chunks.append(
                Chunk(
                    chunk_id=f"flat_{doc_id}_{chunk_num}",
                    text=chunk_text,
                    doc_id=doc_id,
                    tokens=chunk_tokens,
                    token_count=len(chunk_tokens),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    chunk_type="child",
                    parent_id=None,
                    children_ids=[],
                    metadata=chunk_metadata
                )
            )

            chunk_num += 1

            if end_idx >= total_tokens:
                break

            start_idx += step

        print(f"Created {len(chunks)} flat chunks")

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
