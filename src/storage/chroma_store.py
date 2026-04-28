"""
ChromaDB persistent vector storage.

The active storage format uses one collection: ``chunks``.
"""

import os
from typing import Any, Dict, List

from src.models.chunk import Chunk
from src.utils.retrieval_debug import format_ranked_chunk_line


class ChromaVectorStore:
    """Persistent vector storage using a single flat chunk collection."""

    COLLECTION_NAME = "chunks"

    def __init__(self, persist_directory: str = "data/chroma_db"):
        import chromadb
        from chromadb.config import Settings

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._init_collection()

        print(f"ChromaDB initialized: {persist_directory}")
        print(f"   Collection: {self.COLLECTION_NAME}")

    def _init_collection(self) -> None:
        """Initialize or get the flat chunk collection."""
        try:
            self.collection = self.client.get_collection(self.COLLECTION_NAME)
            print(f"   Loaded existing chunks collection ({self.collection.count()} vectors)")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Flat chunks for retrieval"},
            )
            print("   Created new chunks collection")

    def add_chunks(self, chunks: List[Chunk], filename: str = "unknown") -> None:
        """Add flat chunks with filename metadata."""
        print("\nAdding chunks to ChromaDB...")

        if not chunks:
            return

        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "token_count": chunk.token_count,
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "filename": filename,
                }
            )
            metadatas.append(self._clean_metadata(metadata))

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search flat chunks by vector similarity."""
        print(f"\nSearching ChromaDB (top_k={top_k})...")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            print("   No results found")
            return []

        print(f"   Found {len(results['ids'][0])} results")

        formatted_results = []
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = self._clean_metadata(dict(results["metadatas"][0][i] or {}))
            metadata.setdefault("source", "vector")
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "score": similarity,
                    "metadata": metadata,
                }
            )
            print(format_ranked_chunk_line(i + 1, formatted_results[-1]))

        return formatted_results

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch flat chunks by id without running vector similarity search."""
        if not chunk_ids:
            return []

        ordered_ids = list(dict.fromkeys(chunk_id for chunk_id in chunk_ids if chunk_id))
        results = self.collection.get(ids=ordered_ids, include=["documents", "metadatas"])
        if not results.get("ids"):
            return []

        by_id = {}
        for i, chunk_id in enumerate(results["ids"]):
            by_id[chunk_id] = {
                "text": results["documents"][i],
                "metadata": results["metadatas"][i] or {},
            }

        formatted_results = []
        for rank, chunk_id in enumerate(ordered_ids, 1):
            if chunk_id not in by_id:
                continue

            metadata = self._clean_metadata(dict(by_id[chunk_id]["metadata"]))
            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": by_id[chunk_id]["text"],
                    "score": 1.0 / rank,
                    "metadata": metadata,
                }
            )

        return formatted_results

    def delete_document_chunks(self, filename: str) -> int:
        """Delete all chunks whose metadata filename matches the given name."""
        if not filename:
            return 0

        response = self.collection.get(where={"filename": filename})
        ids = response.get("ids") or []
        if not ids:
            return 0

        self.collection.delete(ids=ids)
        print(f"Deleted {len(ids)} chunk(s) for filename={filename}")
        return len(ids)

    def clear_all(self) -> None:
        """Clear all collections."""
        for name in [self.COLLECTION_NAME]:
            if self._get_collection(name) is not None:
                self.client.delete_collection(name)
        self._init_collection()
        print("All collections cleared")

    def _get_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception:
            return None

    def close(self) -> None:
        """Release Chroma's process-local handles."""
        if hasattr(self.client, "clear_system_cache"):
            self.client.clear_system_cache()

    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        total_chunks = self.collection.count()
        return {
            "total_chunks": total_chunks,
            "total_vectors": total_chunks,
        }

    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove hierarchy fields and normalize values for Chroma metadata."""
        cleaned = {}
        for key, value in metadata.items():
            if key in {
                "parent_id",
                "parent_chunk_id",
                "child_chunk_ids",
                "child_chunk_id",
                "chunk_type",
                "matched_child_id",
            }:
                continue
            if value is None:
                continue
            cleaned[key] = value
        return cleaned
