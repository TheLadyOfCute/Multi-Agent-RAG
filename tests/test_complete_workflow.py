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


def test_retry_expansion_updates_sub_query_plans() -> None:
    workflow = CompleteAgenticRAGWorkflow.__new__(CompleteAgenticRAGWorkflow)
    workflow.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    state = AgentState(
        query="hello",
        selected_retrievers=["vector"],
        retriever_quotas={"vector": 4},
        sub_query_plans=[
            {"query": "hello", "retrievers": ["vector"], "quotas": {"vector": 4}},
            {"query": "world", "retrievers": ["keyword"], "quotas": {"keyword": 6}},
        ],
    )

    workflow._expand_retrievers_for_retry(state)

    assert state.selected_retrievers == ["vector", "keyword", "graph"]
    assert state.retriever_quotas == {"vector": 4, "keyword": 10, "graph": 10}
    assert [p["retrievers"] for p in state.sub_query_plans] == [
        ["vector", "keyword", "graph"],
        ["vector", "keyword", "graph"],
    ]
    assert state.sub_query_plans[0]["quotas"] == {
        "vector": 4,
        "keyword": 10,
        "graph": 10,
    }
    assert state.sub_query_plans[1]["quotas"] == {
        "vector": 4,
        "keyword": 6,
        "graph": 10,
    }
    assert (
        state.metadata["retry_retrieval_expansion"]["reason"]
        == "validator_retrieve_more"
    )
