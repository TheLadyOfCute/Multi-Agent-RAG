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
    from src.agents.synthesis import SynthesisAgent
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
    )

    vector_agent = VectorSearchAgent(vector_store=vector_store, embedder=embedder)
    keyword_agent = KeywordSearchAgent(vector_store=vector_store, bm25_index=bm25_index)
    graph_agent = (
        GraphSearchAgent(knowledge_graph=knowledge_graph, vector_store=vector_store)
        if knowledge_graph
        else None
    )

    return CompleteAgenticRAGWorkflow(
        planner=PlannerAgent(llm=llm),
        decomposer=QueryDecomposer(),
        coordinator=RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=keyword_agent,
            graph_agent=graph_agent,
        ),
        validator=ValidatorAgent(llm=llm),
        synthesis=SynthesisAgent(),
        reranker=RerankerAgent(top_k=settings.retrieval_top_k),
        writer=WriterAgent(llm=llm),
        critic=CriticAgent(llm=llm, quality_threshold=0.7),
    )


def load_persisted_knowledge_graph() -> Optional[Any]:
    try:
        from src.config import get_settings
        from src.graph.neo4j_graph_store import Neo4jGraphStore

        settings = get_settings()
        graph = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        return None if graph.is_empty() else graph
    except Exception:
        return None


def create_cli_workflow() -> "CompleteAgenticRAGWorkflow":
    # Direct callers build the workflow from persisted stores.
    from src.cache.redis_cache import RedisCacheService
    from src.ingestion.embedder import EmbeddingGenerator
    from src.storage.chroma_store import ChromaVectorStore
    from src.utils.persistence_restore import restore_or_rebuild_bm25

    vector_store = ChromaVectorStore(persist_directory="data/chroma_db")
    embedder = EmbeddingGenerator(cache_service=RedisCacheService())
    bm25_index = restore_or_rebuild_bm25(vector_store, index_path="data/bm25_index.pkl")
    knowledge_graph = load_persisted_knowledge_graph()
    return create_full_rag_workflow(
        vector_store=vector_store,
        embedder=embedder,
        bm25_index=bm25_index,
        knowledge_graph=knowledge_graph,
    )
