from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    replacements = {
        "src.evaluation.ragas_evaluator": types.SimpleNamespace(
            METRIC_NAMES=(
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ),
            RAGASEvaluator=object,
        ),
        "src.utils.logger": types.SimpleNamespace(
            setup_logger=lambda *args, **kwargs: None,
        ),
    }
    originals = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    path = ROOT / "src" / "agents" / "ragas_evaluation_agent.py"
    spec = importlib.util.spec_from_file_location("test_ragas_agent_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


agent_module = load_module()
RagasEvaluationAgent = agent_module.RagasEvaluationAgent
load_test_questions = agent_module.load_test_questions
read_jsonl = agent_module.read_jsonl


class DummyLogger:
    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def warning(self, message: str, *args: Any) -> None:
        self.warnings.append(message % args if args else message)

    def error(self, message: str, *args: Any) -> None:
        self.errors.append(message % args if args else message)


class FakeWorkflow:
    def run(self, question: str) -> Any:
        chunk = SimpleNamespace(
            text=f"context for {question}",
            chunk_id="chunk-fallback",
            metadata={},
        )
        return {
            "answer": f"answer for {question}",
            "chunks": [chunk],
            "sub_query_plans": [
                {"query": question, "retrievers": ["vector"], "quotas": {"vector": 10}},
            ],
            "retrieval_round": 1,
            "validation_score": 0.8,
            "critic_score": 0.9,
            "metadata": {"trace_id": "trace-1"},
        }


class FakeEvaluator:
    def evaluate_single_case(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.5,
            "context_recall": 0.4,
            "overall": 0.6,
            "metric_errors": {},
        }


def write_test_questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "What is RAG?",
                    "ground_truth": "A retrieval augmented generation system.",
                    "reference_contexts": ["RAG combines retrieval and generation."],
                    "gold_chunk_ids": ["gold-1"],
                    "question_type": "factual",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_collect_rag_outputs_writes_standardized_retrieved_chunk_ids(tmp_path: Path) -> None:
    rag_outputs_path = tmp_path / "rag_outputs.jsonl"
    test_file = tmp_path / "test_questions.json"
    write_test_questions(test_file)

    agent = RagasEvaluationAgent(
        workflow=FakeWorkflow(),
        evaluator=FakeEvaluator(),
        rag_outputs_path=rag_outputs_path,
        logger=DummyLogger(),
    )

    rows = agent.collect_rag_outputs(test_file)

    assert rows[0]["ground_truth"] == "A retrieval augmented generation system."
    assert rows[0]["contexts"] == ["context for What is RAG?"]
    assert rows[0]["retrieved_chunk_ids"] == ["chunk-fallback"]

    cached_rows = read_jsonl(rag_outputs_path)
    assert cached_rows[0]["retrieved_chunk_ids"] == ["chunk-fallback"]
    assert cached_rows[0]["contexts"] == ["context for What is RAG?"]


def test_run_exports_final_metrics_without_double_counting_overall(tmp_path: Path) -> None:
    rag_outputs_path = tmp_path / "rag_outputs.jsonl"
    output_dir = tmp_path / "outputs"
    test_file = tmp_path / "test_questions.json"
    write_test_questions(test_file)

    agent = RagasEvaluationAgent(
        workflow=FakeWorkflow(),
        evaluator=FakeEvaluator(),
        rag_outputs_path=rag_outputs_path,
        logger=DummyLogger(),
    )

    result = agent.run(test_file=test_file, output_dir=output_dir, run_id="run-001")

    score_row = result["scores"][0]
    assert score_row["answer_relevancy"] == 0.7
    assert score_row["overall"] == 0.6

    with open(result["csv_path"], "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        csv_row = next(reader)

    assert "answer_relevancy" in fieldnames
    assert csv_row["answer_relevancy"] == "0.7"


def test_reuse_old_rag_outputs_missing_retrieved_chunk_ids_is_compatible(tmp_path: Path) -> None:
    rag_outputs_path = tmp_path / "rag_outputs.jsonl"
    output_dir = tmp_path / "outputs"
    logger = DummyLogger()

    legacy_row = {
        "id": "q001",
        "question": "What is RAG?",
        "question_type": "factual",
        "reference": "A retrieval augmented generation system.",
        "reference_contexts": ["RAG combines retrieval and generation."],
        "gold_chunk_ids": ["gold-1"],
        "response": "answer",
        "retrieved_contexts": ["legacy context"],
        "retrieval_metadata": {},
        "error": None,
    }
    rag_outputs_path.write_text(
        json.dumps(legacy_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    agent = RagasEvaluationAgent(
        workflow=FakeWorkflow(),
        evaluator=FakeEvaluator(),
        rag_outputs_path=rag_outputs_path,
        logger=logger,
    )

    result = agent.run(
        test_file=tmp_path / "unused.json",
        output_dir=output_dir,
        reuse_rag_outputs=True,
        run_id="run-legacy",
    )

    score_row = result["scores"][0]
    assert score_row["retrieved_chunk_ids"] == []
    assert score_row["answer_relevancy"] == 0.7
    assert result["compatibility_warnings"]
    assert any("retrieved_chunk_ids" in item for item in result["compatibility_warnings"])
    assert any("retrieved_chunk_ids" in item for item in logger.warnings)


def test_load_test_questions_keeps_ground_truth_compatibility(tmp_path: Path) -> None:
    test_file = tmp_path / "test_questions.json"
    write_test_questions(test_file)

    items = load_test_questions(test_file)

    assert items[0]["ground_truth"] == "A retrieval augmented generation system."
    assert items[0]["reference"] == "A retrieval augmented generation system."
