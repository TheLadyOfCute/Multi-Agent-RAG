"""
Retrieval Coordinator Agent - Tactical Level 2 Agent.

Manages retrieval swarm (Vector, Keyword, Graph agents).
Coordinates retrieval, aggregates results, and deduplicates chunks.

Swarm Pattern:
- Spawns multiple retrieval agents
- Each agent uses different retrieval method
- Aggregates all results
- Deduplicates by content similarity
- Returns top-k unique chunks
"""

from typing import List, Dict, Any

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import RetrievalError
from src.config import get_settings
from src.utils.retrieval_debug import chunk_dedup_key


class RetrievalCoordinator(BaseAgent):

    def __init__(
        self,
        vector_agent: BaseAgent = None,
        keyword_agent: BaseAgent = None,
        graph_agent: BaseAgent = None,
        top_k: int = None
    ):
        super().__init__(name="retrieval_coordinator", version="1.0.0")
        
        self.vector_agent = vector_agent
        self.keyword_agent = keyword_agent
        self.graph_agent = graph_agent
        
        # Load settings from config
        settings = get_settings()
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        
        self.log(
            f"Initialized with top_k={self.top_k}",
            level="debug"
        )
    
    def execute(self, state: AgentState) -> AgentState:

        try:
            current_round = state.retrieval_round

            self.log(
                f"Starting retrieval round {current_round} | "
                f"sub_query_plans={len(state.sub_query_plans or [])} plans | "
                f"query={state.query[:50]}...",
                level="info"
            )


            all_results = self._retrieve_by_sub_query_plans(state)
            path_used   = "sub-query-plans"
            retriever_results = dict(state.metadata.get("retriever_results", {}))

            self.log(
                f"Retrieved {len(all_results)} total chunks (path={path_used})",
                level="info"
            )

            # Deduplicate — keep highest-scored copy per content hash
            unique_chunks = self._deduplicate(all_results)
            self.log(
                f"Deduplication: {len(all_results)} → {len(unique_chunks)} unique chunks",
                level="info"
            )

            state.chunks = unique_chunks
            state.retrieval_round = current_round + 1

            state.metadata["retrieval_coordinator"] = {
                "round":            current_round,
                "path":             path_used,
                "sub_queries_used": len(state.sub_queries) if state.sub_queries else 1,
                "sub_query_plans":  len(state.sub_query_plans) if state.sub_query_plans else 0,
                "total_retrieved":  len(all_results),
                "unique_chunks":    len(unique_chunks),
                "retriever_results": retriever_results,
            }

            return state

        except Exception as e:
            self.log(f"Retrieval coordination failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="coordination",
                message=f"Failed to coordinate retrieval: {str(e)}",
                details={"query": state.query, "round": state.retrieval_round}
            ) from e
    
    # ── Per-sub-query plans retrieval (v3 primary path) ──────────────

    def _retrieve_by_sub_query_plans(self, state: AgentState) -> List[Chunk]:
        
        agent_map: Dict[str, Any] = {
            "vector":  self.vector_agent,
            "keyword": self.keyword_agent,
            "graph":   self.graph_agent,
        }

        all_results:      List[Chunk]     = []
        retriever_results: Dict[str, int] = {}
        plans = state.sub_query_plans or []

        for plan_idx, plan in enumerate(plans):
            sub_query  = plan.get("query", state.query)
            retrievers = plan.get("retrievers", ["vector"])
            quotas     = plan.get("quotas", {})

            self.log(
                f"── Sub-query {plan_idx + 1}/{len(plans)} "
                f"| retrievers={retrievers}\n"
                f"   \"{sub_query}\"",
                level="info",
            )

            for name in retrievers:
                agent = agent_map.get(name)
                quota = quotas.get(name, self.top_k)

                if agent is None:
                    self.log(
                        f"  [{plan_idx + 1}] '{name}' not available, skipping",
                        level="warning",
                    )
                    continue

                try:
                    chunks = agent.search_async(sub_query, top_k=quota)
                    for c in chunks:
                        c.metadata["retriever"]      = name
                        c.metadata.setdefault("source", name)
                        c.metadata["sub_query"]      = sub_query
                        c.metadata["sub_query_idx"]  = plan_idx
                        c.metadata["query_used"]     = sub_query
                    all_results.extend(chunks)
                    retriever_results[name] = (
                        retriever_results.get(name, 0) + len(chunks)
                    )
                    self.log(
                        f"  [{plan_idx + 1}] {name.upper()} → {len(chunks)} chunks (top_k={quota})",
                        level="info",
                    )
                except Exception as exc:
                    self.log(
                        f"  [{plan_idx + 1}] {name.upper()} failed: {exc}",
                        level="warning",
                    )

        state.metadata["retriever_results"] = retriever_results
        return all_results

    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:

        if not chunks:
            return []

        # 按 chunk_id 去重，保留每个 id 中分数最高的 chunk，
        # 同时合并所有命中的检索来源（如 "vector|keyword"）
        seen: dict[str, Chunk] = {}
        seen_sources: dict[str, set[str]] = {}

        for chunk in chunks:
            key = chunk_dedup_key(chunk)
            source = (
                chunk.metadata.get("retriever")
                or chunk.metadata.get("source")
                or "unknown"
            )

            if key not in seen:
                seen[key] = chunk
                seen_sources[key] = {source}
            else:
                seen_sources[key].add(source)
                if (chunk.score or 0.0) > (seen[key].score or 0.0):
                    seen[key] = chunk

        # 将合并后的来源写入保留 chunk 的 metadata
        for key, chunk in seen.items():
            sources = sorted(seen_sources[key])
            merged = "|".join(sources)
            chunk.metadata["retriever"] = merged
            chunk.metadata["source"] = merged
            chunk.metadata["all_retrievers"] = sources

        return list(seen.values())
    
    def _select_top_k(self, chunks: List[Chunk], k: int) -> List[Chunk]:
        
        if not chunks:
            return []
        
        # Sort by score (descending)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.score if c.score is not None else 0.0,
            reverse=True
        )
        
        # Return top-k
        return sorted_chunks[:k]
    
