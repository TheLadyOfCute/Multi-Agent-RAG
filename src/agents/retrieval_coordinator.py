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
from collections import defaultdict

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk, Strategy
from src.utils.exceptions import RetrievalError
from src.config import get_settings
from src.utils.retrieval_debug import chunk_dedup_key, merge_retriever_sources


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

            # ── Path 1: Per-sub-query plans ─────────────
            if state.sub_query_plans:
                all_results = self._retrieve_by_sub_query_plans(state)
                path_used   = "sub-query-plans"
                retriever_results = dict(
                    state.metadata.get("retriever_results", {})
                )

            # ── Path 2: Planner-selected retrievers  ───────
            elif state.selected_retrievers:
                all_results = self._retrieve_by_planner_selection(state)
                path_used   = "planner-selection"
                retriever_results = dict(
                    state.metadata.get("retriever_results", {})
                )

            # ── Path 3: Legacy strategy-based routing (fallback) ──────
            else:
                strategy = state.strategy
                self.log(
                    f"No retriever selection from planner, "
                    f"falling back to strategy={strategy}",
                    level="warning"
                )
                if strategy == Strategy.SIMPLE or strategy == "simple":
                    all_results = self._retrieve_simple(state.query)
                else:
                    queries = state.sub_queries or [state.query]
                    all_results = self._retrieve_multihop(state.query, queries)
                path_used = f"strategy-{strategy}"
                retriever_results = self._count_by_retriever(all_results)
            # ──────────────────────────────────────────────────────────

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

            # NOTE: We no longer apply top-k here.
            # The full unique pool is passed to SynthesisAgent → RerankerAgent
            # which will apply top-k after Cohere reranking.
            state.chunks          = unique_chunks
            state.retrieval_round = current_round + 1

            state.metadata["retrieval_coordinator"] = {
                "round":            current_round,
                "path":             path_used,
                "selected_retrievers": list(state.selected_retrievers),
                "retriever_quotas": dict(state.retriever_quotas),
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

    # ── Planner-driven retrieval (v2 compat path) ─────────────────────

    def _retrieve_by_planner_selection(self, state: AgentState) -> List[Chunk]:
        """
        Execute only the retrievers chosen by PlannerAgent, each with
        its own quota.

        For every name in ``state.selected_retrievers`` the
        corresponding agent is called with ``top_k = retriever_quotas[name]``.
        Results are tagged with ``metadata["retriever"]`` for
        traceability and merged into a single flat list.

        If a retriever agent is not available (e.g. graph agent not
        initialised because no KG was built) the entry is skipped with
        a warning so the remaining retrievers can still contribute.

        Args:
            state: AgentState carrying selected_retrievers and
                   retriever_quotas populated by PlannerAgent.

        Returns:
            Flat list of Chunk objects from all executed retrievers.
        """
        agent_map: Dict[str, Any] = {
            "vector":  self.vector_agent,
            "keyword": self.keyword_agent,
            "graph":   self.graph_agent,
        }

        all_results: List[Chunk] = []
        retriever_results: Dict[str, int] = {}

        queries = state.sub_queries or [state.query]
        for query_idx, query in enumerate(queries):
            for name in state.selected_retrievers:
                agent = agent_map.get(name)
                quota = state.retriever_quotas.get(name, self.top_k)

                if agent is None:
                    self.log(
                        f"[PLANNER] Retriever '{name}' not available, skipping",
                        level="warning",
                    )
                    continue

                try:
                    chunks = agent.search_async(query, top_k=quota)
                    for c in chunks:
                        c.metadata["retriever"] = name
                        c.metadata.setdefault("source", name)
                        c.metadata["sub_query"] = query
                        c.metadata["sub_query_idx"] = query_idx
                        c.metadata["query_used"] = query
                    all_results.extend(chunks)
                    retriever_results[name] = retriever_results.get(name, 0) + len(chunks)
                    self.log(
                        f"[PLANNER] q{query_idx + 1}/{len(queries)} "
                        f"{name.upper()} -> {len(chunks)} chunks "
                        f"(top_k={quota})",
                        level="info",
                    )
                except Exception as exc:
                    self.log(
                        f"[PLANNER] Retriever '{name}' failed for q{query_idx + 1}: {exc}",
                        level="warning",
                    )

        state.metadata["retriever_results"] = retriever_results
        return all_results

        for name in state.selected_retrievers:
            agent = agent_map.get(name)
            quota = state.retriever_quotas.get(name, self.top_k)

            if agent is None:
                self.log(
                    f"[PLANNER] Retriever '{name}' not available, skipping",
                    level="warning"
                )
                continue

            try:
                chunks = agent.search_async(state.query, top_k=quota)
                # Tag each chunk with the retriever that produced it
                for c in chunks:
                    c.metadata["retriever"] = name
                    c.metadata.setdefault("source", name)
                all_results.extend(chunks)
                retriever_results[name] = len(chunks)
                self.log(
                    f"[PLANNER] {name.upper()} → {len(chunks)} chunks "
                    f"(top_k={quota})",
                    level="info"
                )
            except Exception as exc:
                self.log(
                    f"[PLANNER] Retriever '{name}' failed: {exc}",
                    level="warning"
                )

        state.metadata["retriever_results"] = retriever_results
        return all_results

    def _count_by_retriever(self, chunks: List[Chunk]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for chunk in chunks:
            source = (
                chunk.metadata.get("retriever")
                or chunk.metadata.get("source")
                or "unknown"
            )
            counts[source] += 1
        return dict(counts)

    # ── Strategy-driven retrieval (legacy fallback paths) ─────────────

    def _retrieve_simple(self, query: str) -> List[Chunk]:
        """
        简单查询：仅使用向量检索，跳过关键词和图检索。
        """
        self.log(f"[SIMPLE] 向量检索: {query[:50]}", level="info")
        
        if not self.vector_agent:
            self.log("向量检索 agent 不可用，降级为全量 swarm", level="warning")
            return self._spawn_swarm(query)
        
        try:
            results = self.vector_agent.search_async(query, top_k=self.top_k)
            self.log(f"[SIMPLE] 向量检索返回 {len(results)} 个块", level="info")
            return results
        except Exception as e:
            self.log(f"[SIMPLE] 向量检索失败: {e}，降级为全量 swarm", level="warning")
            return self._spawn_swarm(query)
    
    def _retrieve_multihop(self, original_query: str, sub_queries: List[str]) -> List[Chunk]:
        """
        复杂查询：对每个 sub_query 分别执行向量+关键词检索，结果汇总。
        每个子查询 top_k 适当缩小，避免结果集过大。
        """
        self.log(
            f"[MULTIHOP] 并行检索 {len(sub_queries)} 个子查询",
            level="info"
        )
        
        per_query_k = max(3, self.top_k // max(len(sub_queries), 1))
        all_results: List[Chunk] = []
        
        for i, q in enumerate(sub_queries):
            self.log(f"[MULTIHOP] 子查询 {i+1}/{len(sub_queries)}: {q[:50]}", level="debug")
            
            # 向量检索
            if self.vector_agent:
                try:
                    chunks = self.vector_agent.search_async(q, top_k=per_query_k)
                    # 记录来源子查询
                    for c in chunks:
                        c.metadata["sub_query"] = q
                        c.metadata["sub_query_idx"] = i
                    all_results.extend(chunks)
                except Exception as e:
                    self.log(f"[MULTIHOP] 向量检索子查询 {i+1} 失败: {e}", level="warning")
            
            # 关键词检索
            if self.keyword_agent:
                try:
                    chunks = self.keyword_agent.search_async(q, top_k=per_query_k)
                    for c in chunks:
                        c.metadata["sub_query"] = q
                        c.metadata["sub_query_idx"] = i
                    all_results.extend(chunks)
                except Exception as e:
                    self.log(f"[MULTIHOP] 关键词检索子查询 {i+1} 失败: {e}", level="warning")
        
        # 同时用原始查询补充一次，保证整体相关性
        if self.vector_agent:
            try:
                extra = self.vector_agent.search_async(original_query, top_k=self.top_k)
                all_results.extend(extra)
            except Exception:
                pass
        
        self.log(f"[MULTIHOP] 总计检索 {len(all_results)} 个块", level="info")
        return all_results
    
    def _spawn_swarm(self, query: str) -> List[Chunk]:
        """Spawn retrieval swarm (private method)."""
        
        self.log(f"Spawning retrieval swarm for: {query}")
        
        # Collect available agents
        agents = []
        
        if self.vector_agent:
            agents.append(('vector', self.vector_agent))
        
        if self.keyword_agent:
            agents.append(('keyword', self.keyword_agent))
        
        if self.graph_agent:  # ← Just check if exists!
            agents.append(('graph', self.graph_agent))
            self.log("Graph search agent included in swarm")
        else:
            self.log("Graph search unavailable", level="warning")
        
        # Execute agents
        all_results = []
        
        for agent_name, agent in agents:
            self.log(f"Executing {agent_name} agent...")
            try:
                results = agent.search_async(query, top_k=self.top_k)
                self.log(f"{agent_name}: {len(results)} chunks")
                all_results.extend(results)
            except Exception as e:
                self.log(f"{agent_name} failed: {e}", level="error")
        
        self.log(f"Swarm complete: {len(all_results)} chunks from {len(agents)} agents")
        
        return all_results
    
    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove duplicate chunks based on content similarity.
        
        Uses text hashing to identify duplicates. Keeps chunk with
        highest score when duplicates found.
        
        Args:
            chunks: List of chunks (may contain duplicates)
        
        Returns:
            List of unique chunks
        
        Example:
            >>> duplicates = [chunk1, chunk1_copy, chunk2]
            >>> unique = coordinator._deduplicate(duplicates)
            >>> print(len(unique))  # 2
        """
        if not chunks:
            return []
        
        # Group by stable chunk identity first.
        hash_groups = defaultdict(list)
        
        for chunk in chunks:
            content_hash = chunk_dedup_key(chunk)
            hash_groups[content_hash].append(chunk)
        
        # Keep best chunk from each group
        unique_chunks = []
        for group in hash_groups.values():
            # Sort by score (descending)
            sorted_group = sorted(
                group,
                key=lambda c: c.score if c.score is not None else 0.0,
                reverse=True
            )
            # Keep highest scored
            best = sorted_group[0]
            merge_retriever_sources(best, sorted_group[1:])
            unique_chunks.append(best)
        
        return unique_chunks
    
    def _select_top_k(self, chunks: List[Chunk], k: int) -> List[Chunk]:
        """
        Select top-k chunks by score.
        
        Args:
            chunks: List of chunks
            k: Number to select
        
        Returns:
            Top-k chunks sorted by score (descending)
        
        Example:
            >>> top_10 = coordinator._select_top_k(chunks, 10)
            >>> print(len(top_10))  # 10
            >>> print(top_10[0].score >= top_10[-1].score)  # True
        """
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
    
