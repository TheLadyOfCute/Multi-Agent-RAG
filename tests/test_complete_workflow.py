from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.agent_state import AgentState
from src.workflows.complete_workflow import CompleteAgenticRAGWorkflow


def make_workflow_with_stubbed_invoke(raw_result):
    workflow = CompleteAgenticRAGWorkflow.__new__(CompleteAgenticRAGWorkflow)
    workflow.logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    workflow.workflow = SimpleNamespace(invoke=lambda state: raw_result)
    return workflow


def test_run_returns_dict_when_langgraph_returns_dict() -> None:
    raw_result = {
        "answer": "ok",
        "chunks": [],
        "selected_retrievers": ["vector"],
        "retrieval_round": 1,
        "critic_score": 0.9,
        "metadata": {"regeneration_count": 0},
    }
    workflow = make_workflow_with_stubbed_invoke(raw_result)

    result = workflow.run("hello")

    assert isinstance(result, dict)
    assert result["answer"] == "ok"
    assert result["selected_retrievers"] == ["vector"]


def test_run_normalizes_agent_state_to_dict() -> None:
    raw_result = AgentState(
        query="hello",
        answer="ok",
        selected_retrievers=["keyword"],
        retrieval_round=2,
        metadata={"regeneration_count": 1},
    )
    workflow = make_workflow_with_stubbed_invoke(raw_result)

    result = workflow.run("hello")

    assert isinstance(result, dict)
    assert result["query"] == "hello"
    assert result["answer"] == "ok"
    assert result["selected_retrievers"] == ["keyword"]
    assert result["metadata"]["regeneration_count"] == 1
