"""Evaluation-related application use cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.server.utils.state import RuntimeState
from src.graph.neo4j_helpers import close_neo4j_store, open_neo4j_store
from src.storage.factory import close_vector_store, open_vector_store
from src.utils.logger import setup_logger

DEFAULT_TEST_FILE = Path("data/test_questions.json")
UPLOADED_TEST_FILE = Path("data/evaluations/uploaded_test_questions.json")

logger = setup_logger("evaluation_use_cases")


class LoadTestQuestionsUseCase:
    def execute(self, path: Path = DEFAULT_TEST_FILE) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "question" in data:
            items = [data]
        elif isinstance(data, dict) and "questions" in data:
            items = [{"id": f"q{i + 1:03d}", "question": q} for i, q in enumerate(data.get("questions") or [])]
        else:
            raise ValueError("test_questions.json must be an array, one object, or {'questions': [...]}")

        normalized = []
        for idx, item in enumerate(items, start=1):
            if isinstance(item, str):
                item = {"question": item}
            question = str(item.get("question", "")).strip()
            if not question:
                raise ValueError(f"Question #{idx} is missing 'question'")
            normalized.append(
                {
                    "id": str(item.get("id") or f"q{idx:03d}"),
                    "question": question,
                    "question_type": str(item.get("question_type") or ""),
                    "reference": str(item.get("reference") or item.get("ground_truth") or ""),
                }
            )
        return normalized


class SaveTestQuestionsUploadUseCase:
    def __init__(self, load_test_questions_use_case: LoadTestQuestionsUseCase):
        self.load_test_questions_use_case = load_test_questions_use_case

    def execute(self, content: bytes) -> dict[str, Any]:
        UPLOADED_TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        UPLOADED_TEST_FILE.write_bytes(content)
        questions = self.load_test_questions_use_case.execute(UPLOADED_TEST_FILE)
        return {"test_file": str(UPLOADED_TEST_FILE), "questions": questions}


class SubmitRagasEvaluationUseCase:
    def __init__(self, runtime_state: RuntimeState, task_registry, embedder: Any):
        self.runtime_state = runtime_state
        self.task_registry = task_registry
        self.embedder = embedder

    def submit(self, test_file: str, reuse_rag_outputs: bool = False) -> str:
        # 提交阶段只负责创建后台任务和更新运行态，
        # 真正的评测执行交给后台线程。
        with self.runtime_state.lock:
            self.runtime_state.ragas_evaluation_running = True

        def runner(task_id: str, update_task: Callable[..., None]) -> dict[str, Any]:
            try:
                return self.run(task_id, update_task, test_file, reuse_rag_outputs)
            finally:
                with self.runtime_state.lock:
                    self.runtime_state.eval_task_id = None
                    self.runtime_state.ragas_evaluation_running = False

        task_id = self.task_registry.submit_task("ragas_eval", runner)
        with self.runtime_state.lock:
            self.runtime_state.eval_task_id = task_id
        return task_id

    def run(self, task_id: str, update_task: Callable[..., None], test_file: str, reuse_rag_outputs: bool) -> dict[str, Any]:
        from src.agents.ragas_evaluation_agent import RagasEvaluationAgent

        def progress_callback(current: int, total: int, stage: str, row: dict) -> None:
            update_task(
                task_id,
                progress=current / max(total, 1),
                current=current,
                total=total,
                stage=stage,
                last_id=row.get("id", ""),
            )

        # 复用已有 RAG 输出时，直接进入评估，跳过向量库、BM25、图谱、工作流的初始化。
        if reuse_rag_outputs:
            update_task(task_id, progress=0.1, stage="loading_cached_outputs", last_id="reading cached RAG outputs")
            return RagasEvaluationAgent(workflow=None).run(
                test_file=test_file,
                output_dir="outputs/evaluations",
                reuse_rag_outputs=True,
                progress_callback=progress_callback,
            )

        from src.utils.persistence_restore import restore_or_rebuild_bm25

        update_task(task_id, progress=0.03, stage="opening_vector_store", last_id="opening Chroma vector store")
        vector_store = open_vector_store()
        knowledge_graph = None
        try:
            update_task(task_id, progress=0.08, stage="loading_bm25", last_id="loading or rebuilding BM25 index")
            bm25_index = None
            try:
                bm25_index = restore_or_rebuild_bm25(vector_store, index_path="data/bm25_index.pkl")
            except Exception as exc:
                logger.warning(f"BM25 load skipped for RAGAS: {exc}")

            update_task(task_id, progress=0.14, stage="loading_graph", last_id="checking Neo4j graph availability")
            try:
                knowledge_graph = open_neo4j_store()
                if knowledge_graph.is_empty():
                    close_neo4j_store(knowledge_graph)
                    knowledge_graph = None
            except Exception as exc:
                logger.warning(f"Neo4j load skipped for RAGAS: {exc}")
                knowledge_graph = None

            update_task(task_id, progress=0.2, stage="initializing_workflow", last_id="initializing multi-agent workflow")
            from src.workflows.factory import create_full_rag_workflow

            workflow = create_full_rag_workflow(
                vector_store=vector_store,
                embedder=self.embedder,
                bm25_index=bm25_index,
                knowledge_graph=knowledge_graph,
            )

            update_task(task_id, progress=0.25, stage="running_ragas", last_id="running RAGAS evaluation")
            return RagasEvaluationAgent(workflow=workflow).run(
                test_file=test_file,
                output_dir="outputs/evaluations",
                reuse_rag_outputs=False,
                progress_callback=progress_callback,
            )
        finally:
            close_vector_store(vector_store)
            close_neo4j_store(knowledge_graph)
