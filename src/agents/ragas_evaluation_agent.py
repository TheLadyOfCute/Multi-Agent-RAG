"""Agent for running full-RAG outputs through RAGAS evaluation."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.evaluation.ragas_evaluator import METRIC_NAMES, RAGASEvaluator
from src.utils.logger import setup_logger

DEFAULT_RAG_OUTPUTS_PATH = Path("data/evaluations/rag_outputs.jsonl")


def load_test_questions(path: str | Path) -> List[Dict[str, Any]]:
    """Load target test questions and normalize optional fields."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "question" in data:
        items = [data]
    elif isinstance(data, dict) and "questions" in data:
        items = [
            {"id": f"q{i + 1:03d}", "question": question}
            for i, question in enumerate(data.get("questions") or [])
        ]
    else:
        raise ValueError(
            "test_questions.json must be an array, one object, or {'questions': [...]}"
        )

    normalized = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"question": item}
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"Question #{index} is missing 'question'")
        normalized.append(
            {
                "id": str(item.get("id") or f"q{index:03d}"),
                "question": question,
                "question_type": str(item.get("question_type") or ""),
                "ground_truth": str(item.get("ground_truth") or ""),
                "reference": str(item.get("reference") or item.get("ground_truth") or ""),
                "reference_contexts": list(item.get("reference_contexts") or []),
                "gold_chunk_ids": [str(v) for v in item.get("gold_chunk_ids") or []],
            }
        )
    return normalized


class RagasEvaluationAgent:
    """Run RAG once, cache outputs, then evaluate cached rows with RAGAS."""

    def __init__(
        self,
        workflow: Any,
        evaluator: Optional[RAGASEvaluator] = None,
        rag_outputs_path: str | Path = DEFAULT_RAG_OUTPUTS_PATH,
        logger: Any = None,
    ):
        self.workflow = workflow
        self.evaluator = evaluator or RAGASEvaluator()
        self.rag_outputs_path = Path(rag_outputs_path)
        self.logger = logger or setup_logger("ragas_evaluation_agent", level="INFO")

    def run(
        self,
        test_file: str | Path,
        output_dir: str | Path = "outputs/evaluations",
        reuse_rag_outputs: bool = False,
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / run_id
        output_path.mkdir(parents=True, exist_ok=True)

        if reuse_rag_outputs:
            rag_rows = read_jsonl(self.rag_outputs_path)
            rag_rows, compatibility_warnings = self._normalize_cached_rows(rag_rows)
        else:
            rag_rows = self.collect_rag_outputs(test_file, progress_callback)
            compatibility_warnings = []

        scores = self.evaluate_outputs(rag_rows, progress_callback)
        summary = build_summary(run_id, scores)

        scores_path = output_path / "ragas_scores.jsonl"
        summary_path = output_path / "summary.json"
        csv_path = output_path / "ragas_scores.csv"

        write_jsonl(scores_path, scores)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_scores_csv(csv_path, scores)

        return {
            "run_id": run_id,
            "run_dir": str(output_path),
            "rag_outputs_path": str(self.rag_outputs_path),
            "scores_path": str(scores_path),
            "summary_path": str(summary_path),
            "csv_path": str(csv_path),
            "scores": scores,
            "summary": summary,
            "compatibility_warnings": compatibility_warnings,
        }

    def collect_rag_outputs(
        self,
        test_file: str | Path,
        progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        questions = load_test_questions(test_file)
        rows = []
        total = len(questions)
        for index, item in enumerate(questions, start=1):
            row = self._run_one_question(item)
            rows.append(row)
            if progress_callback:
                progress_callback(index, total, "rag", row)
        self.rag_outputs_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.rag_outputs_path, rows)
        return rows

    def _run_one_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "id": item["id"],
            "question": item["question"],
            "question_type": item.get("question_type", ""),
            "ground_truth": item.get("ground_truth",""),
            "reference_contexts": item.get("reference_contexts", []),
            "gold_chunk_ids": item.get("gold_chunk_ids", []),
            "response": "",
            "contexts": [],
            "retrieved_chunk_ids": [],
            "retrieval_metadata": {},
            "error": None,
        }
        try:
            state = self.workflow.run(item["question"])
            chunks = list(state.get("chunks") or [])
            base.update(
                {
                    "response": str(state.get("answer") or ""),
                    "contexts": [getattr(chunk, "text", "") for chunk in chunks],
                    "retrieved_chunk_ids": [chunk_id_for_evaluation(chunk) for chunk in chunks],
                    "retrieval_metadata": {
                        "selected_retrievers": state.get("selected_retrievers", []),
                        "retriever_quotas": state.get("retriever_quotas", {}),
                        "retrieval_round": state.get("retrieval_round"),
                        "validation_score": state.get("validation_score"),
                        "critic_score": state.get("critic_score"),
                        "metadata": state.get("metadata", {}),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - per-row errors are required output
            base["error"] = str(exc)
            self.logger.error("RAG failed for %s: %s", item["id"], exc)
        return base

    def evaluate_outputs(
        self,
        rag_rows: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        total = len(rag_rows)
        score_rows = []
        for index, row in enumerate(rag_rows, start=1):
            score_row = self._evaluate_one_row(row)
            score_rows.append(score_row)
            if progress_callback:
                progress_callback(index, total, "ragas", score_row)
        return score_rows

    def _evaluate_one_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        output = {
            "id": row.get("id"),
            "question": row.get("question"),
            "question_type": row.get("question_type", ""),
            "response": row.get("response", ""),
            "ground_truth": row.get("ground_truth"),
            "retrieved_chunk_ids": row.get("retrieved_chunk_ids", []),
            "gold_chunk_ids": row.get("gold_chunk_ids", []),
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "overall": None,
            "error": row.get("error"),
        }
        if row.get("error"):
            return output

        try:
            scores = self.evaluator.evaluate_single_case(
                question=row.get("question", ""),
                answer=row.get("response", ""),
                contexts=row.get("contexts", row.get("retrieved_contexts", [])),
                ground_truth=row.get("ground_truth"),
            )
            for name in METRIC_NAMES:
                output[name] = scores.get(name)
            output["overall"] = mean_metric(output[name] for name in METRIC_NAMES)
        except Exception as exc:  # noqa: BLE001 - keep evaluating later rows
            output["error"] = str(exc)
            self.logger.error("RAGAS failed for %s: %s", row.get("id"), exc)
        return output

    def _normalize_cached_rows(
        self,
        rag_rows: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        normalized_rows = []
        warnings_list = []
        for row in rag_rows:
            normalized_row = dict(row)
            normalized_row["ground_truth"] = row.get("ground_truth","")
            normalized_row["contexts"] = row.get("contexts", row.get("retrieved_contexts", []))
            if "retrieved_chunk_ids" not in row:
                warning = (
                    f"Cached rag output {row.get('id', '')} is missing retrieved_chunk_ids; "
                    "reusing the row with an empty retrieved_chunk_ids list."
                )
                warnings_list.append(warning)
                if self.logger:
                    self.logger.warning(warning)
            normalized_row["retrieved_chunk_ids"] = list(row.get("retrieved_chunk_ids") or [])
            normalized_rows.append(normalized_row)
        return normalized_rows, warnings_list


def chunk_id_for_evaluation(chunk: Any) -> str:
    return str(getattr(chunk, "chunk_id", ""))


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RAG outputs cache not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_scores_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "question",
        "question_type",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "overall",
        "retrieved_chunk_ids",
        "gold_chunk_ids",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: json.dumps(row.get(column), ensure_ascii=False)
                    if isinstance(row.get(column), list)
                    else row.get(column)
                    for column in columns
                }
            )


def build_summary(run_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "run_id": run_id,
        "num_questions": len(rows),
        "num_success": sum(1 for row in rows if not row.get("error")),
        "num_failed": sum(1 for row in rows if row.get("error")),
    }
    for metric in METRIC_NAMES:
        summary[f"mean_{metric}"] = mean_metric(row.get(metric) for row in rows)
    summary["mean_overall"] = mean_metric(row.get("overall") for row in rows)
    return summary


def mean_metric(values: Iterable[Any]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)
