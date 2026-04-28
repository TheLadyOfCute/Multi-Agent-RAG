"""
Vector Search Agent - Operational Level 3 Agent.

Performs semantic search using vector embeddings.
Part of retrieval swarm coordinated by RetrievalCoordinator.
"""

from typing import List

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.storage.chroma_store import ChromaVectorStore
from src.ingestion.embedder import EmbeddingGenerator
from src.utils.exceptions import AgentExecutionError


class VectorSearchAgent(BaseAgent):
    def __init__(self, vector_store: ChromaVectorStore, embedder: EmbeddingGenerator):
        super().__init__(name="vector_search", version="1.0.0")
        self.vector_store = vector_store
        self.embedder = embedder
        self.log("Initialized with ChromaDB vector store", level="debug")

    def execute(self, state: AgentState) -> AgentState:
        try:
            query = state.query
            self.log(f"Vector search for: {query[:50]}...", level="info")
            query_embedding = self.embedder.generate_query_embedding(query)
            results = self.vector_store.search(query_embedding=query_embedding, top_k=10)

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
                        "source": "vector",
                        **result_metadata,
                    },
                )
                chunks.append(chunk)

            state.chunks = chunks
            self.log(f"Vector search retrieved {len(chunks)} chunks", level="info")
            return state
        except Exception as e:
            self.log(f"Vector search failed: {str(e)}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Vector search failed: {str(e)}",
                details={"query": state.query},
            ) from e

    def search_async(self, query: str, top_k: int = 10) -> List[Chunk]:
        try:
            query_embedding = self.embedder.generate_query_embedding(query)
            results = self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
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
                        "source": "vector",
                    },
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            self.log(f"Async vector search failed: {str(e)}", level="error")
            return []
