"""Factory helpers for constructing RAG workflows."""

from __future__ import annotations

from typing import Any, Optional


def create_full_rag_workflow(
    vector_store: Any,
    embedder: Any,
    bm25_index: Any = None,
    knowledge_graph: Any = None,
) -> "CompleteAgenticRAGWorkflow":
    # This factory only wires agents into a runnable workflow.
    from langchain_openai import ChatOpenAI
    from src.agents.critic import CriticAgent
    from src.agents.planner import PlannerAgent
    from src.agents.query_decomposer import QueryDecomposer
    from src.agents.reranker import RerankerAgent
    from src.agents.retrieval_coordinator import RetrievalCoordinator
    from src.agents.validator import ValidatorAgent
    from src.agents.writer import WriterAgent
    from src.config import get_settings
    from src.retrieval.graph_search import GraphSearchAgent
    from src.retrieval.keyword_search import KeywordSearchAgent
    from src.retrieval.vector_search import VectorSearchAgent
    from src.workflows.complete_workflow import CompleteAgenticRAGWorkflow

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        extra_body={"enable_thinking": False},
    )

    vector_agent = VectorSearchAgent(vector_store=vector_store, embedder=embedder)
    keyword_agent = KeywordSearchAgent(vector_store=vector_store, bm25_index=bm25_index)
    graph_agent = GraphSearchAgent(knowledge_graph=knowledge_graph, vector_store=vector_store)

    return CompleteAgenticRAGWorkflow(
        planner=PlannerAgent(llm=llm),
        decomposer=QueryDecomposer(llm=llm),
        coordinator=RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=keyword_agent,
            graph_agent=graph_agent,
        ),
        validator=ValidatorAgent(llm=llm),
        reranker=RerankerAgent(top_k=settings.retrieval_top_k),
        writer=WriterAgent(llm=llm),
        critic=CriticAgent(llm=llm, quality_threshold=settings.critic_threshold),
        vector_store=vector_store,
    )
