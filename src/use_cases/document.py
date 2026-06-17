"""Document-related application use cases."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

from src.server.utils.paths import BM25_INDEX_PATH, UPLOAD_DIR
from src.server.utils.state import RuntimeState
from src.cache.redis_cache import RedisCacheService
from src.graph.neo4j_helpers import (
    close_neo4j_store,
    get_neo4j_stats,
    open_neo4j_store,
    refresh_neo4j_stats_best_effort,
)
from src.storage.factory import close_vector_store, open_vector_store
from src.utils.logger import setup_logger

logger = setup_logger("document_use_cases")


def open_bm25_index(index_path: str):
    from src.retrieval.bm25_index import BM25Index

    return BM25Index(index_path=index_path)


class RestoreDocumentStateUseCase:
    def __init__(self, runtime_state: RuntimeState):
        # 保存运行时状态实例，用于读写系统状态
        self.runtime_state = runtime_state

    def restore_persisted_state(self) -> None:
        # 启动时恢复持久化状态，目标是尽快恢复前端可见状态
        self._restore_from_persisted_store(update_empty_status=True)

    def ensure_persisted_state_loaded(self) -> None:
        # 加锁检查当前状态，避免并发读取导致状态不一致
        with self.runtime_state.lock:
            # 判断 RAG 是否已初始化且已有文档数据
            already_loaded = self.runtime_state.rag_initialized and bool(self.runtime_state.documents)

            # 获取当前是否正在处理任务
            processing = self.runtime_state.processing

        # 如果状态已加载或正在处理任务，则无需重复恢复
        if already_loaded or processing:
            return

        # 按需从持久化存储恢复状态
        self._restore_from_persisted_store(update_empty_status=False)

    def _restore_from_persisted_store(self, *, update_empty_status: bool) -> None:
        # 初始化向量库对象，便于 finally 中统一关闭
        vector_store = None

        try:
            # 打开持久化向量库
            vector_store = open_vector_store()

            # 获取向量库统计信息
            stats = vector_store.get_stats()

            # 如果向量库为空，则只刷新图数据库状态并按需更新恢复状态
            if stats["total_vectors"] <= 0:
                # 尝试刷新 Neo4j 统计信息，不阻断主流程
                refresh_neo4j_stats_best_effort(self.runtime_state)

                # 启动恢复时更新空库状态提示
                if update_empty_status:
                    # 加锁更新运行时状态
                    with self.runtime_state.lock:
                        # 标记 RAG 尚未初始化
                        self.runtime_state.rag_initialized = False

                        # 记录向量库为空的恢复状态
                        self.runtime_state.restore_status = "ChromaDB is empty; waiting for document upload."

                # 向量库为空时直接结束恢复流程
                return

            # 延迟导入持久化恢复工具，避免启动阶段不必要依赖
            from src.utils.persistence_restore import build_document_records, restore_or_rebuild_bm25

            # 根据持久化向量库重建前端可见的文档记录
            documents = build_document_records(vector_store)

            try:
                # 尝试恢复或重建 BM25 索引
                restore_or_rebuild_bm25(vector_store, index_path=str(BM25_INDEX_PATH))
            except Exception as exc:
                # BM25 恢复失败时记录警告，但不影响主恢复流程
                logger.warning(f"Could not restore BM25 index: {exc}")

            # 尝试刷新 Neo4j 统计信息
            refresh_neo4j_stats_best_effort(self.runtime_state)

            # 加锁写入恢复后的运行时状态
            with self.runtime_state.lock:
                # 更新文档列表
                self.runtime_state.documents = documents

                # 标记 RAG 已完成初始化
                self.runtime_state.rag_initialized = True

                # 记录恢复结果摘要
                self.runtime_state.restore_status = f"Restored {stats['total_chunks']} chunks, {len(documents)} documents."

        except Exception as exc:
            # 恢复异常时仍尝试刷新 Neo4j 统计信息
            refresh_neo4j_stats_best_effort(self.runtime_state)

            # 加锁记录恢复失败状态
            with self.runtime_state.lock:
                # 保存恢复失败原因
                self.runtime_state.restore_status = f"Persisted restore failed: {exc}"

                # 标记 RAG 未初始化
                self.runtime_state.rag_initialized = False

        finally:
            # 无论恢复是否成功，都关闭向量库连接
            close_vector_store(vector_store)


class EnsurePersistedStateLoadedUseCase:
    def __init__(self, runtime_state: RuntimeState):
        self.restore_use_case = RestoreDocumentStateUseCase(runtime_state)

    def execute(self) -> None:
        self.restore_use_case.ensure_persisted_state_loaded()


class ListDocumentsUseCase:
    def __init__(self, runtime_state: RuntimeState):
        self.runtime_state = runtime_state
        self.ensure_loaded = EnsurePersistedStateLoadedUseCase(runtime_state)

    def execute(self) -> list[dict[str, Any]]:
        self.ensure_loaded.execute()
        with self.runtime_state.lock:
            return [dict(doc) for doc in self.runtime_state.documents]


class SaveUploadUseCase:
    def execute(self, filename: str, file_obj: BinaryIO) -> Path:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / Path(filename).name
        with target.open("wb") as handle:
            while True:
                chunk = file_obj.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return target


class ProcessUploadedDocumentUseCase:
    def __init__(self, runtime_state: RuntimeState, embedder: Any, cache_service: RedisCacheService):
        self.runtime_state = runtime_state
        self.embedder = embedder
        self.cache_service = cache_service

    def execute(self, file_path: Path, update_task=None, task_id: str | None = None) -> dict[str, Any]:
        """Process one uploaded file end-to-end.

        这是“文档进入系统”的主入口：
        load -> chunk -> embed -> graph -> vector store -> bm25 -> runtime state
        """

        vector_store = None
        filename = file_path.name

        def progress(value: float, stage: str, message: str = "") -> None:
            if update_task and task_id:
                update_task(task_id, progress=value, stage=stage, last_id=message or filename)

        with self.runtime_state.lock:
            self.runtime_state.processing = True

        try:
            from src.ingestion.document_loader import DocumentLoader

            file_ext = file_path.suffix.upper()
            loader = DocumentLoader()
            vector_store = open_vector_store()

            progress(0.1, "initialize", "Initializing RAG components")
            progress(0.25, "load", f"Loading {filename}")
            doc = loader.load(str(file_path))
            progress(0.35, "chunk_advice", "Choosing chunk parameters")
            chunk_size, chunk_overlap = self._resolve_chunk_params(doc.text)
            progress(0.45, "chunk", f"Chunking {filename}")
            chunks = self._build_chunks(doc, filename, file_ext, chunk_size, chunk_overlap)
            progress(0.6, "embedding", f"{len(chunks)} chunks")

            # chunk 才能进入向量检索链路。
            self._embed_chunks(chunks, self.embedder)
            progress(0.72, "graph", "Building graph")
            graph_available = self._build_graph_best_effort(chunks)
            progress(0.85, "store", "Writing vectors")
            vector_store.add_chunks(chunks, filename=filename)
            progress(0.92, "bm25", "Building BM25")
            bm25_available = self._build_bm25_best_effort(vector_store)
            document = self._build_document_record(
                filename=filename,
                file_path=file_path,
                file_ext=file_ext,
                page_count=loader.count_pages(str(file_path)),
                chunk_count=len(chunks),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            # 新增或更新运行时文档状态，并标记 RAG 已初始化
            self._upsert_document_state(document)
            self.cache_service.clear_answer_cache()
            progress(1.0, "done", filename)
            return {
                "document": document,
                "chunks": len(chunks),
                "graph_available": graph_available,
                "bm25_available": bm25_available,
            }
        finally:
            close_vector_store(vector_store)
            with self.runtime_state.lock:
                self.runtime_state.processing = False

    @staticmethod
    def _advise_chunk_params(text: str) -> dict[str, int]:
        try:
            from langchain_openai import ChatOpenAI

            from src.agents.chunking_advisor import ChunkingAdvisorAgent
            from src.config import get_settings

            settings = get_settings()
            llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0.0,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
                extra_body={"enable_thinking": False},
            )
            return ChunkingAdvisorAgent(llm=llm).advise(text)
        except Exception as exc:
            logger.warning(f"ChunkingAdvisor failed: {exc}; using defaults")
            from src.config import get_settings

            settings = get_settings()
            return {"chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}

    def _resolve_chunk_params(self, text: str) -> tuple[int, int]:
        advised_params = self._advise_chunk_params(text)
        return int(advised_params["chunk_size"]), int(advised_params["chunk_overlap"])

    @staticmethod
    def _build_chunks(doc: Any, filename: str, file_ext: str, chunk_size: int, chunk_overlap: int) -> list[Any]:
        from src.ingestion.flat_chunker import FlatChunker

        chunk_metadata = {
            "filename": filename,
            "file_type": file_ext,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            **doc.metadata,
        }
        chunker = FlatChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunker.chunk_text(text=doc.text, doc_id=doc.doc_id, metadata=chunk_metadata)

    @staticmethod
    def _embed_chunks(chunks: list[Any], embedder: Any) -> None:
        if not chunks:
            return
        
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.generate(texts)#批量embedding生成，内部会优先复用缓存
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    def _build_graph_best_effort(self, chunks: list[Any]) -> bool:
        try:
            from src.graph.entity_extractor import EntityExtractor
            from src.graph.relationship_extractor import RelationshipExtractor

            entity_extractor = EntityExtractor()
            rel_extractor = RelationshipExtractor()

            chunk_entities: dict[str, list] = {}
            chunk_relationships: dict[str, list] = {}
            for chunk in chunks:
                entities = entity_extractor.extract(chunk.text)
                chunk_entities[chunk.chunk_id] = entities
                if len(entities) >= 2:
                    chunk_relationships[chunk.chunk_id] = rel_extractor.extract_from_sentence(chunk.text, entities)
                else:
                    chunk_relationships[chunk.chunk_id] = []

            graph_store = open_neo4j_store()
            try:
                graph_store.build_from_chunks(chunks, chunk_entities, chunk_relationships)
            finally:
                graph_store.close()

            try:
                get_neo4j_stats(self.runtime_state)
            except Exception as exc:
                logger.warning(f"Could not refresh Neo4j stats: {exc}")
            return True
        except Exception as exc:
            logger.warning(f"Graph building failed: {exc}")
            return False

    @staticmethod
    def _build_bm25_best_effort(vector_store: Any) -> bool:
        try:
            from src.retrieval.bm25_index import BM25Index

            if BM25_INDEX_PATH.exists():
                BM25_INDEX_PATH.unlink()
            bm25 = BM25Index(index_path=str(BM25_INDEX_PATH))
            bm25.build_from_vector_store(vector_store)
            bm25.save()
            return True
        except Exception as exc:
            logger.warning(f"BM25 build failed: {exc}")
            return False

    @staticmethod
    def _build_document_record(
        *,
        filename: str,
        file_path: Path,
        file_ext: str,
        page_count: int,
        chunk_count: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, Any]:
        return {
            "name": filename,
            "path": str(file_path),
            "type": file_ext,
            "pages": page_count,
            "chunks": chunk_count,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "restored": False,
        }

    def _upsert_document_state(self, document: dict[str, Any]) -> None:
        with self.runtime_state.lock:
            existing = next((doc for doc in self.runtime_state.documents if doc["name"] == document["name"]), None)
            if existing:
                existing.update(document)
            else:
                self.runtime_state.documents.append(document)
            self.runtime_state.rag_initialized = True


class DeleteDocumentUseCase:
    def __init__(self, runtime_state: RuntimeState, cache_service: RedisCacheService):
        self.runtime_state = runtime_state
        self.cache_service = cache_service

    def execute(self, name: str) -> dict[str, Any]:
        decoded = Path(name).name
        removed = None
        with self.runtime_state.lock:
            kept = []
            for doc in self.runtime_state.documents:
                if doc.get("name") == decoded:
                    removed = doc
                else:
                    kept.append(doc)
            self.runtime_state.documents = kept

        file_deleted = False
        vector_chunks_deleted = False
        deleted_chunks_count = 0
        chunk_ids: list[str] = []
        graph_chunks_deleted = False
        graph_available = False
        bm25_rebuilt = False
        bm25_deleted = False
        remaining_chunks: int | None = None
        if removed:
            doc_path = removed.get("path") or ""
            if doc_path and os.path.exists(doc_path):
                os.remove(doc_path)
                file_deleted = True

            vector_store = None
            try:
                vector_store = open_vector_store()
                chunk_ids = vector_store.get_document_chunk_ids(decoded)
                deleted_chunks_count = vector_store.delete_document_chunks(decoded)
                vector_chunks_deleted = deleted_chunks_count > 0
                remaining_chunks = vector_store.get_stats()["total_chunks"]
                bm25_rebuilt, bm25_deleted = self._refresh_bm25_after_delete(vector_store, remaining_chunks)
            except Exception as exc:
                logger.warning(f"Delete vector chunks/BM25 refresh failed for '{decoded}': {exc}")
            finally:
                close_vector_store(vector_store)

            graph_chunks_deleted, graph_available = self._delete_graph_chunks_best_effort(chunk_ids, decoded)
            refresh_neo4j_stats_best_effort(self.runtime_state)

            with self.runtime_state.lock:
                if not self.runtime_state.documents or remaining_chunks == 0:
                    self.runtime_state.rag_initialized = False

            self.cache_service.clear_answer_cache()

        return {
            "deleted": removed is not None,
            "file_deleted": file_deleted,
            "vector_chunks_deleted": vector_chunks_deleted,
            "vector_chunks_count": deleted_chunks_count,
            "graph_chunks_deleted": graph_chunks_deleted,
            "graph_available": graph_available,
            "bm25_rebuilt": bm25_rebuilt,
            "bm25_deleted": bm25_deleted,
        }

    @staticmethod
    def _refresh_bm25_after_delete(vector_store: Any, remaining_chunks: int) -> tuple[bool, bool]:
        if remaining_chunks <= 0:
            if BM25_INDEX_PATH.exists():
                BM25_INDEX_PATH.unlink()
                return False, True
            return False, False

        bm25 = open_bm25_index(index_path=str(BM25_INDEX_PATH))
        bm25.build_from_vector_store(vector_store)
        bm25.save()
        return True, False

    @staticmethod
    def _delete_graph_chunks_best_effort(chunk_ids: list[str], filename: str) -> tuple[bool, bool]:
        if not chunk_ids:
            return False, False

        graph_store = None
        try:
            graph_store = open_neo4j_store()
            graph_store.delete_chunks(chunk_ids)
            return True, True
        except Exception as exc:
            logger.warning(f"Delete graph chunks failed for '{filename}': {exc}")
            return False, False
        finally:
            close_neo4j_store(graph_store)


class ClearAllDataUseCase:
    def __init__(self, runtime_state: RuntimeState, task_registry, cache_service: RedisCacheService):
        self.runtime_state = runtime_state
        self.task_registry = task_registry
        self.cache_service = cache_service

    def execute(self) -> dict[str, Any]:
        # 清空动作同时影响磁盘存储和内存态，所以两边都要一起重置。
        vector_store = None
        try:
            vector_store = open_vector_store()
            vector_store.clear_all()
        finally:
            close_vector_store(vector_store)

        for path in (BM25_INDEX_PATH,):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

        self.cache_service.clear_answer_cache()
        self.runtime_state.reset_runtime()
        self.task_registry.reset()
        return {"cleared": True}


class PreviewDocumentUseCase:
    def __init__(self, runtime_state: RuntimeState, list_documents_use_case: ListDocumentsUseCase):
        self.runtime_state = runtime_state
        self.list_documents_use_case = list_documents_use_case

    def execute(self, name: str, max_chars: int = 8000) -> dict[str, Any]:
        # 预览时重新读取文件，保证预览内容和当前磁盘文件一致。
        document = next((doc for doc in self.list_documents_use_case.execute() if doc.get("name") == name), None)
        if not document:
            raise FileNotFoundError(name)
        path = document.get("path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(path or name)
        from src.ingestion.document_loader import DocumentLoader

        loader = DocumentLoader()
        loaded = loader.load(path)
        text = loaded.text[:max_chars]
        return {"name": name, "text": text, "chars": len(loaded.text), "words": len(loaded.text.split())}
