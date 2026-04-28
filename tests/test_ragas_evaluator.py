from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    replacements = {
        "langchain_openai": types.SimpleNamespace(
            ChatOpenAI=object,
            OpenAIEmbeddings=object,
        ),
        "datasets": types.SimpleNamespace(
            Dataset=types.SimpleNamespace(from_dict=lambda data: data),
        ),
        "ragas": types.SimpleNamespace(
            evaluate=lambda *args, **kwargs: {},
        ),
        "ragas.llms": types.SimpleNamespace(LangchainLLMWrapper=object),
        "ragas.metrics": types.SimpleNamespace(
            answer_relevancy=object(),
            context_precision=object(),
            context_recall=object(),
            faithfulness=object(),
        ),
        "ragas.run_config": types.SimpleNamespace(
            RunConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ),
        "src.config": types.SimpleNamespace(
            get_settings=lambda: types.SimpleNamespace(),
        ),
    }
    originals = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    path = ROOT / "src" / "evaluation" / "ragas_evaluator.py"
    spec = importlib.util.spec_from_file_location("test_ragas_evaluator_module", path)
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


evaluator_module = load_module()
RAGASEvaluator = evaluator_module.RAGASEvaluator


def make_evaluator() -> RAGASEvaluator:
    return RAGASEvaluator(llm=object(), embeddings=object())


def install_fake_ragas(monkeypatch: pytest.MonkeyPatch, results: List[Any]) -> None:
    queue = list(results)

    def fake_evaluate(*args, **kwargs):
        assert queue, "unexpected extra RAGAS evaluate() call"
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(evaluator_module, "evaluate", fake_evaluate)


def test_evaluate_single_case_returns_normalized_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ragas(
        monkeypatch,
        [
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.4,
                "context_precision": 0.7,
                "context_recall": 0.6,
            },
        ],
    )
    evaluator = make_evaluator()

    scores = evaluator.evaluate_single_case(
        question="Why?",
        answer="Because.",
        contexts=["ctx"],
        ground_truth="gt",
    )

    assert scores["faithfulness"] == 0.8
    assert scores["context_precision"] == 0.7
    assert scores["context_recall"] == 0.6
    assert scores["answer_relevancy"] == 0.4
    assert scores["overall"] == pytest.approx((0.8 + 0.4 + 0.7 + 0.6) / 4)
    assert scores["metric_errors"] == {}


def test_evaluate_single_case_supports_response_relevancy_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ragas(
        monkeypatch,
        [
            {
                "faithfulness": 0.8,
                "response_relevancy": 0.45,
                "context_precision": 0.7,
                "context_recall": 0.6,
            },
        ],
    )
    evaluator = make_evaluator()

    scores = evaluator.evaluate_single_case(
        question="Why?",
        answer="Because.",
        contexts=["ctx"],
        ground_truth="gt",
    )

    assert scores["answer_relevancy"] == 0.45


def test_evaluate_single_case_returns_none_when_ragas_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ragas(
        monkeypatch,
        [
            RuntimeError("ragas failed"),
        ],
    )

    evaluator = make_evaluator()

    scores = evaluator.evaluate_single_case(
        question="Why?",
        answer="Because.",
        contexts=["ctx"],
        ground_truth="gt",
    )

    assert scores["faithfulness"] is None
    assert scores["answer_relevancy"] is None
    assert scores["context_precision"] is None
    assert scores["context_recall"] is None
    assert scores["overall"] is None
    assert set(scores["metric_errors"]) == {"official_metrics"}


def test_evaluate_rag_system_aggregates_only_final_metric_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = make_evaluator()

    per_case: List[Dict[str, Any]] = [
        {
            "faithfulness": 1.0,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.0,
            "overall": 0.5,
            "metric_errors": {},
        },
        {
            "faithfulness": 0.0,
            "answer_relevancy": 1.0,
            "context_precision": 0.5,
            "context_recall": 1.0,
            "overall": 0.625,
            "metric_errors": {},
        },
    ]

    def fake_single_case(**_: Any) -> Dict[str, Any]:
        return per_case.pop(0)

    monkeypatch.setattr(evaluator, "evaluate_single_case", fake_single_case)

    scores = evaluator.evaluate_rag_system(
        questions=["q1", "q2"],
        answers=["a1", "a2"],
        contexts=[["c1"], ["c2"]],
        ground_truths=["g1", "g2"],
    )

    assert scores["faithfulness"] == 0.5
    assert scores["answer_relevancy"] == 0.75
    assert scores["context_precision"] == 0.5
    assert scores["context_recall"] == 0.5
    assert scores["overall"] == pytest.approx((0.5 + 0.75 + 0.5 + 0.5) / 4)


def test_module_loads_without_langchain_private_api_module() -> None:
    sys.modules.pop("langchain_core._api", None)
    original_langchain_openai = sys.modules.get("langchain_openai")
    original_src_config = sys.modules.get("src.config")
    sys.modules["langchain_openai"] = types.SimpleNamespace(
        ChatOpenAI=object,
        OpenAIEmbeddings=object,
    )
    sys.modules["src.config"] = types.SimpleNamespace(
        get_settings=lambda: types.SimpleNamespace(),
    )

    path = ROOT / "src" / "evaluation" / "ragas_evaluator.py"
    spec = importlib.util.spec_from_file_location(
        "test_ragas_evaluator_module_without_private_api",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if original_langchain_openai is None:
            sys.modules.pop("langchain_openai", None)
        else:
            sys.modules["langchain_openai"] = original_langchain_openai
        if original_src_config is None:
            sys.modules.pop("src.config", None)
        else:
            sys.modules["src.config"] = original_src_config

    assert issubclass(module.LangChainDeprecationWarning, Warning)
