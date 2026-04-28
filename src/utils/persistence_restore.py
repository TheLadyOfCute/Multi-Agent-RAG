"""Helpers for restoring persisted app state after an API restart."""

from pathlib import Path
from typing import Any, Dict, List, Optional


def build_document_records(vector_store: Any) -> List[Dict[str, Any]]:
    """Build sidebar document records from persisted Chroma metadata."""

    # �?Chroma 集合中读取已持久化的元数�?    records = vector_store.collection.get(include=["metadatas"])

    # 用文件名聚合文档记录
    documents: Dict[str, Dict[str, Any]] = {}

    # 遍历所有元数据记录
    for meta in records.get("metadatas", []) or []:
        # 获取当前文本块所属文件名
        filename = meta.get("filename", "")

        # 如果文件名为空，则跳过该记录
        if not filename:
            continue

        # 获取或创建当前文件对应的文档记录
        doc = documents.setdefault(filename, _new_document_record(filename, meta))

        # 累加当前文档的文本块数量
        doc["chunks"] += 1

    # 返回聚合后的文档记录列表
    return list(documents.values())


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

    # 延迟导入 BM25 索引，避免模块加载时产生不必要依�?    from src.retrieval.bm25_index import BM25Index

    # 获取 Chroma 中当前文本块数量，作�?BM25 是否匹配的依�?    expected_chunks = vector_store.collection.count()

    # 创建 BM25 索引实例，并指定索引文件路径
    bm25_index = BM25Index(index_path=index_path)

    # 获取当前 BM25 索引统计信息
    stats = bm25_index.get_stats()

    # 如果已有索引且文本块数量一致，则直接复用该索引
    if stats.get("built") and stats.get("total_chunks") == expected_chunks:
        return bm25_index

    # 当索引不存在或数量不匹配时，从向量库重建 BM25 索引
    bm25_index.build_from_vector_store(vector_store)

    # 将重建后�?BM25 索引保存到磁�?    bm25_index.save()

    # 返回可用�?BM25 索引实例
    return bm25_index


def _new_document_record(filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    file_type = (("." + filename.rsplit(".", 1)[-1]).upper() if "." in filename else ".TXT")
    restored_path = _resolve_document_path(filename, metadata)
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


def _resolve_document_path(filename: str, metadata: Dict[str, Any]) -> str:
    """Recover a usable local file path for document preview/delete actions."""
    candidate_path = metadata.get("file_path") or metadata.get("path")
    if isinstance(candidate_path, str) and candidate_path.strip():
        return candidate_path

    fallback = Path("data/uploads") / filename
    if fallback.exists():
        return str(fallback)

    return ""
