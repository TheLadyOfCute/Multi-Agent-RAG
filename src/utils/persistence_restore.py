"""Helpers for restoring persisted app state after an API restart."""

from pathlib import Path
from typing import Any, Dict, List, Optional


def build_document_records(vector_store: Any) -> List[Dict[str, Any]]:
    """Build sidebar document records from persisted Chroma metadata."""
    records = vector_store.collection.get(include=["metadatas"])

    documents: Dict[str, Dict[str, Any]] = {}
    for raw_meta in records.get("metadatas", []) or []:
        meta = raw_meta or {}
        filename = _extract_filename(meta)
        if not filename:
            continue

        doc = documents.setdefault(filename, _new_document_record(filename, meta))
        doc["chunks"] += 1

    if documents:
        return list(documents.values())

    # Backward-compatible fallback for old vector rows with missing filename metadata.
    uploads_dir = Path("data/uploads")
    if not uploads_dir.exists():
        return []

    fallback_docs: List[Dict[str, Any]] = []
    for file_path in sorted(uploads_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            continue
        fallback_docs.append(
            {
                "name": file_path.name,
                "path": str(file_path),
                "type": file_path.suffix.upper() or ".TXT",
                "chunks": 0,
                "chunk_size": "-",
                "chunk_overlap": "-",
                "uploaded_at": "未知",
                "restored": True,
            }
        )
    return fallback_docs


def load_persisted_knowledge_graphs(graph_dir: str = "data/graphs") -> Optional[Any]:
    """Load one or more saved knowledge graph pickle files."""
    graph_path = Path(graph_dir)
    graph_files = sorted(graph_path.glob("*_graph.pkl"))
    if not graph_files:
        return None

    import networkx as nx
    from src.graph.graph_builder import KnowledgeGraph

    merged = KnowledgeGraph()
    for file_path in graph_files:
        kg = KnowledgeGraph()
        kg.load(str(file_path))
        merged.graph = nx.compose(merged.graph, kg.graph)

    merged.entity_count = merged.graph.number_of_nodes()
    merged.relationship_count = merged.graph.number_of_edges()
    return merged if merged.graph.number_of_nodes() > 0 else None


def restore_or_rebuild_bm25(vector_store: Any, index_path: str = "data/bm25_index.pkl") -> Any:
    """Load BM25 if it matches Chroma chunk count, otherwise rebuild it."""
    from src.retrieval.bm25_index import BM25Index

    expected_chunks = vector_store.collection.count()
    bm25_index = BM25Index(index_path=index_path)
    stats = bm25_index.get_stats()

    if stats.get("built") and stats.get("total_chunks") == expected_chunks:
        return bm25_index

    bm25_index.build_from_vector_store(vector_store)
    bm25_index.save()
    return bm25_index


def _extract_filename(metadata: Dict[str, Any]) -> str:
    direct_keys = [
        "filename",
        "file_name",
        "document_name",
        "doc_name",
        "source_file",
        "source_filename",
    ]
    for key in direct_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).name

    path_keys = ["file_path", "path", "source_path", "source", "filepath"]
    for key in path_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).name

    return ""


def _new_document_record(filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    file_type = ("." + filename.rsplit(".", 1)[-1]).upper() if "." in filename else ".TXT"
    restored_path = _best_effort_resolve_path(filename, metadata)
    uploaded_at = metadata.get("created_at") or metadata.get("modified_at")
    if isinstance(uploaded_at, str):
        uploaded_at = uploaded_at.replace("T", " ")[:19]

    return {
        "name": filename,
        "path": restored_path,
        "type": file_type,
        "chunks": 0,
        "chunk_size": metadata.get("chunk_size", "-"),
        "chunk_overlap": metadata.get("chunk_overlap", "-"),
        "uploaded_at": uploaded_at or "未知",
        "restored": True,
    }


def _best_effort_resolve_path(filename: str, metadata: Dict[str, Any]) -> str:
    candidate_keys = ["file_path", "path", "source_path", "source", "filepath"]
    for key in candidate_keys:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            continue

        path = Path(value)
        if path.exists():
            return str(path)

        # Old absolute path may be invalid in a new environment; try basename in uploads.
        if path.name:
            fallback = Path("data/uploads") / path.name
            if fallback.exists():
                return str(fallback)

    fallback = Path("data/uploads") / filename
    if fallback.exists():
        return str(fallback)

    return ""
