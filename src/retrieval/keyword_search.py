"""
Keyword Search Agent - Operational Level 3 Agent.

Performs keyword-based search using BM25 algorithm.
Part of retrieval swarm coordinated by RetrievalCoordinator.
"""

from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.retrieval.bm25_index import BM25Index
from src.storage.chroma_store import ChromaVectorStore
from src.utils.exceptions import AgentExecutionError


class KeywordSearchAgent(BaseAgent):
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        index_path: str = "data/bm25_index.pkl",
        bm25_index: Optional[BM25Index] = None,
    ):
        super().__init__(name="keyword_search", version="1.0.0")
        self.vector_store = vector_store

        if bm25_index is not None:
            self.bm25_index = bm25_index
            self.log("Using injected BM25 index", level="info")
        else:
            self.bm25_index = BM25Index(index_path=index_path)
            if not self.bm25_index.bm25:
                self.log("BM25 index not found, building now...", level="info")
                try:
                    self.bm25_index.build_from_vector_store(vector_store)
                    self.bm25_index.save()
                    self.log("BM25 index build completed", level="info")
                except Exception as e:
                    self.log(f"BM25 index build failed: {e}", level="warning")

        self.log("Initialized with BM25 index", level="debug")

    def execute(self, state: AgentState) -> AgentState:
        try:
            query = state.query
            self.log(f"Keyword search for: {query[:50]}...", level="info")

            if not self.bm25_index.bm25:
                self.log("BM25 index not available, returning empty", level="warning")
                state.chunks = []
                return state

            results = self.bm25_index.search(query, top_k=10)
            chunks = []
            for result in results:
                result_metadata = dict(result.get("metadata", {}) or {})
                chunk = Chunk(
                    text=result["text"],
                    doc_id="unknown",
                    chunk_id=result["chunk_id"],
                    score=result["score"],
                    metadata={
                        "filename": result_metadata.get("filename", "unknown"),
                        "source": "keyword",
                        **result_metadata,
                    },
                )
                chunks.append(chunk)

            state.chunks = chunks
            self.log(f"Keyword search retrieved {len(chunks)} chunks", level="info")
            return state

        except Exception as e:
            self.log(f"Keyword search failed: {str(e)}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Keyword search failed: {str(e)}",
                details={"query": state.query},
            ) from e

    def search_async(self, query: str, top_k: int = 10) -> List[Chunk]:
        try:
            if not self.bm25_index.bm25:
                self.log("BM25 index not available", level="warning")
                return []

            results = self.bm25_index.search(query, top_k=top_k)
            chunks = []
            for result in results:
                result_metadata = dict(result.get("metadata", {}) or {})
                chunk = Chunk(
                    text=result["text"],
                    doc_id="unknown",
                    chunk_id=result["chunk_id"],
                    score=result["score"],
                    metadata={
                        "filename": result_metadata.get("filename", "unknown"),
                        **result_metadata,
                        "source": "keyword",
                    },
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            self.log(f"Async keyword search failed: {str(e)}", level="error")
            return []

    def rebuild_index(self) -> None:
        self.log("Rebuilding BM25 index...", level="info")
        try:
            self.bm25_index.rebuild()
            self.log("BM25 index rebuilt", level="info")
        except Exception as e:
            self.log(f"Failed to rebuild index: {e}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Index rebuild failed: {str(e)}",
                details={},
            ) from e
