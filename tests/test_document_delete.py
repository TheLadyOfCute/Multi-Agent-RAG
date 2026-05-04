from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DummyCache:
    def __init__(self) -> None:
        self.cleared = 0

    def clear_answer_cache(self) -> None:
        self.cleared += 1


class DummyVectorStore:
    def __init__(self, chunk_ids: list[str], remaining_chunks: int) -> None:
        self.chunk_ids = chunk_ids
        self.remaining_chunks = remaining_chunks
        self.deleted_filename = None

    def get_document_chunk_ids(self, filename: str) -> list[str]:
        return self.chunk_ids

    def delete_document_chunks(self, filename: str) -> int:
        self.deleted_filename = filename
        return len(self.chunk_ids)

    def get_stats(self) -> dict[str, int]:
        return {"total_chunks": self.remaining_chunks, "total_vectors": self.remaining_chunks}


class DummyGraphStore:
    def __init__(self) -> None:
        self.deleted_chunk_ids = []
        self.closed = False

    def delete_chunks(self, chunk_ids: list[str]) -> dict[str, int]:
        self.deleted_chunk_ids = chunk_ids
        return {"chunks": len(chunk_ids), "relationships_deleted": 0, "entities_deleted": 0}

    def close(self) -> None:
        self.closed = True


class FakeBM25Index:
    built = 0
    saved = 0

    def __init__(self, index_path: str) -> None:
        self.index_path = index_path

    def build_from_vector_store(self, vector_store) -> None:
        FakeBM25Index.built += 1

    def save(self) -> None:
        FakeBM25Index.saved += 1


def test_delete_document_removes_file_indexes_graph_and_cache(monkeypatch, tmp_path) -> None:
    from src.server.utils.state import RuntimeState
    from src.use_cases import document as document_module

    uploaded = tmp_path / "sample.txt"
    uploaded.write_text("hello", encoding="utf-8")
    bm25_path = tmp_path / "bm25_index.pkl"
    vector_store = DummyVectorStore(chunk_ids=["chunk-1", "chunk-2"], remaining_chunks=3)
    graph_store = DummyGraphStore()
    cache = DummyCache()
    state = RuntimeState()
    state.documents = [{"name": "sample.txt", "path": str(uploaded)}]
    state.rag_initialized = True
    FakeBM25Index.built = 0
    FakeBM25Index.saved = 0

    monkeypatch.setattr(document_module, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(document_module, "open_bm25_index", lambda index_path: FakeBM25Index(index_path))
    monkeypatch.setattr(document_module, "open_vector_store", lambda: vector_store)
    monkeypatch.setattr(document_module, "close_vector_store", lambda store: None)
    monkeypatch.setattr(document_module, "open_neo4j_store", lambda: graph_store)
    monkeypatch.setattr(document_module, "close_neo4j_store", lambda store: store.close())
    monkeypatch.setattr(document_module, "refresh_neo4j_stats_best_effort", lambda state: None)

    result = document_module.DeleteDocumentUseCase(state, cache).execute("sample.txt")

    assert result["deleted"] is True
    assert result["file_deleted"] is True
    assert result["vector_chunks_deleted"] is True
    assert result["vector_chunks_count"] == 2
    assert result["graph_chunks_deleted"] is True
    assert result["graph_available"] is True
    assert result["bm25_rebuilt"] is True
    assert result["bm25_deleted"] is False
    assert uploaded.exists() is False
    assert vector_store.deleted_filename == "sample.txt"
    assert graph_store.deleted_chunk_ids == ["chunk-1", "chunk-2"]
    assert FakeBM25Index.built == 1
    assert FakeBM25Index.saved == 1
    assert cache.cleared == 1
    assert state.documents == []
    assert state.rag_initialized is False


def test_delete_document_removes_stale_bm25_when_chroma_becomes_empty(monkeypatch, tmp_path) -> None:
    from src.server.utils.state import RuntimeState
    from src.use_cases import document as document_module

    uploaded = tmp_path / "last.txt"
    uploaded.write_text("hello", encoding="utf-8")
    bm25_path = tmp_path / "bm25_index.pkl"
    bm25_path.write_bytes(b"stale")
    vector_store = DummyVectorStore(chunk_ids=["chunk-1"], remaining_chunks=0)
    cache = DummyCache()
    state = RuntimeState()
    state.documents = [{"name": "last.txt", "path": str(uploaded)}]
    state.rag_initialized = True
    FakeBM25Index.built = 0
    FakeBM25Index.saved = 0

    monkeypatch.setattr(document_module, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(document_module, "open_bm25_index", lambda index_path: FakeBM25Index(index_path))
    monkeypatch.setattr(document_module, "open_vector_store", lambda: vector_store)
    monkeypatch.setattr(document_module, "close_vector_store", lambda store: None)
    monkeypatch.setattr(document_module, "open_neo4j_store", lambda: DummyGraphStore())
    monkeypatch.setattr(document_module, "close_neo4j_store", lambda store: None)
    monkeypatch.setattr(document_module, "refresh_neo4j_stats_best_effort", lambda state: None)

    result = document_module.DeleteDocumentUseCase(state, cache).execute("last.txt")

    assert result["bm25_rebuilt"] is False
    assert result["bm25_deleted"] is True
    assert bm25_path.exists() is False
    assert FakeBM25Index.built == 0
    assert FakeBM25Index.saved == 0
    assert state.rag_initialized is False


def test_delete_missing_document_does_not_touch_stores(monkeypatch) -> None:
    from src.server.utils.state import RuntimeState
    from src.use_cases import document as document_module

    state = RuntimeState()
    state.documents = [{"name": "other.txt", "path": ""}]
    cache = DummyCache()

    def fail_open_vector_store():
        raise AssertionError("vector store should not open")

    monkeypatch.setattr(document_module, "open_vector_store", fail_open_vector_store)

    result = document_module.DeleteDocumentUseCase(state, cache).execute("missing.txt")

    assert result == {
        "deleted": False,
        "file_deleted": False,
        "vector_chunks_deleted": False,
        "vector_chunks_count": 0,
        "graph_chunks_deleted": False,
        "graph_available": False,
        "bm25_rebuilt": False,
        "bm25_deleted": False,
    }
    assert cache.cleared == 0
    assert state.documents == [{"name": "other.txt", "path": ""}]


def test_delete_document_endpoint_returns_extended_status(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from src.server import main
    from src.server.routes import documents as documents_route

    expected = {
        "deleted": True,
        "file_deleted": True,
        "vector_chunks_deleted": True,
        "vector_chunks_count": 1,
        "graph_chunks_deleted": True,
        "graph_available": True,
        "bm25_rebuilt": True,
        "bm25_deleted": False,
    }

    class FakeDeleteUseCase:
        def execute(self, name: str) -> dict:
            assert name == "sample.txt"
            return expected

    monkeypatch.setattr(documents_route, "get_delete_document_use_case", lambda: FakeDeleteUseCase())

    with TestClient(main.app) as client:
        response = client.delete("/api/documents/sample.txt")

    assert response.status_code == 200
    assert response.json() == expected


def test_neo4j_delete_chunks_removes_evidence_and_prunes_empty_items(monkeypatch) -> None:
    replacements = {
        "spacy": types.SimpleNamespace(load=lambda *args, **kwargs: None),
    }
    originals = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        module = importlib.import_module("src.graph.neo4j_graph_store")
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    class FakeCounters:
        def __init__(self, relationships_deleted=0, nodes_deleted=0) -> None:
            self.relationships_deleted = relationships_deleted
            self.nodes_deleted = nodes_deleted

    class FakeSummary:
        def __init__(self, counters) -> None:
            self.counters = counters

    class FakeResult:
        def __init__(self, counters) -> None:
            self._summary = FakeSummary(counters)

        def consume(self):
            return self._summary

    class FakeSession:
        def __init__(self) -> None:
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query, **params):
            self.calls.append((query, params))
            if "DELETE r" in query:
                return FakeResult(FakeCounters(relationships_deleted=2))
            if "DELETE e" in query:
                return FakeResult(FakeCounters(nodes_deleted=3))
            return FakeResult(FakeCounters())

    class FakeDriver:
        def __init__(self) -> None:
            self.session_obj = FakeSession()

        def session(self):
            return self.session_obj

    driver = FakeDriver()
    store = module.Neo4jGraphStore(driver=driver)

    result = store.delete_chunks(["chunk-1", "", "chunk-1", "chunk-2"])

    assert result == {"chunks": 2, "relationships_deleted": 2, "entities_deleted": 3}
    assert len(driver.session_obj.calls) == 4
    assert driver.session_obj.calls[0][1] == {"chunk_ids": ["chunk-1", "chunk-2"]}
    assert "e.chunk_ids" in driver.session_obj.calls[0][0]
    assert "r.evidence_json" in driver.session_obj.calls[1][0]
    assert "CONTAINS" in driver.session_obj.calls[1][0]
    assert "DELETE r" in driver.session_obj.calls[2][0]
    assert "DELETE e" in driver.session_obj.calls[3][0]
