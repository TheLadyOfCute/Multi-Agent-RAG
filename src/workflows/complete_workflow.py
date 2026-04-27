"""
Complete LangGraph Workflow - All Agents Orchestration (v3).

Pipeline (8 nodes):

    decomposer  ← entry point
      ↓  (LLM decides: decompose or pass-through)
    planner
      ↓  (assigns per-sub-query retrieval plans)
    retrieval  ←─────────────────────────┐
      ↓                                  │ retry
    synthesis                            │
      ↓                                  │
    reranker  (Cohere / weighted-fallback)│
      ↓                                  │
    validator ──(RETRIEVE_MORE)──────────┘
      ↓ PROCEED
    writer  ←─────────────────┐
      ↓                       │ regenerate
    critic ────(APPROVED)───► END

Changes from v2
---------------
- Entry point changed from planner to decomposer.
  Decomposer now makes its own LLM-based decision on whether to split
  the query; it no longer depends on the strategy set by the planner.
- Planner moved after decomposer. It reads state.sub_queries and assigns
  per-sub-query retrieval plans (state.sub_query_plans) in one LLM call.
- Conditional edge on decomposer (simple/decompose) removed.
  Fixed edge decomposer → planner → retrieval replaces it.
- RetrievalCoordinator now uses _retrieve_by_sub_query_plans() as the
  primary path, giving each sub-query its own set of retrievers.
"""

from typing import Dict, Any, Literal, TypedDict

from langgraph.graph import StateGraph, END

from src.models.agent_state import AgentState
from src.agents.planner import PlannerAgent
from src.agents.query_decomposer import QueryDecomposer
from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.validator import ValidatorAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.reranker import RerankerAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent, CriticDecision
from src.utils.logger import setup_logger
from src.utils.workflow_trace import format_stage_trace, summarize_chunks
from src.utils.retrieval_debug import format_ranked_chunk_line
from src.utils.exceptions import OrchestrationError


class CompleteAgenticRAGWorkflow:
    """
    Complete LangGraph workflow for the Agentic RAG system (v3).

    Orchestrates all agents in a multi-stage pipeline:

    Stage 1 — Decomposition + Planning
        Decomposer  → Planner

    Stage 2 — Retrieval + Candidate pool (with validator retry loop)
        Retrieval Coordinator → Synthesis → Reranker → Validator
        (Validator loops back to Retrieval Coordinator if quality gate fails)

    Stage 3 — Generation (writer self-reflection loop)
        Writer ↔ Critic

    Parameters
    ----------
    planner : PlannerAgent
    decomposer : QueryDecomposer
    coordinator : RetrievalCoordinator
    validator : ValidatorAgent
    synthesis : SynthesisAgent
    reranker : RerankerAgent
    writer : WriterAgent
    critic : CriticAgent
    """

    def __init__(
        self,
        planner:     PlannerAgent,
        decomposer:  QueryDecomposer,
        coordinator: RetrievalCoordinator,
        validator:   ValidatorAgent,
        synthesis:   SynthesisAgent,
        reranker:    RerankerAgent,
        writer:      WriterAgent,
        critic:      CriticAgent,
    ):
        self.planner     = planner
        self.decomposer  = decomposer
        self.coordinator = coordinator
        self.validator   = validator
        self.synthesis   = synthesis
        self.reranker    = reranker
        self.writer      = writer
        self.critic      = critic

        self.logger = setup_logger("complete_workflow", level="INFO")
        self.workflow = self._build_workflow()
        self.logger.info(
            "Complete AgenticRAG workflow v3 initialized (8 nodes)"
        )

    # ------------------------------------------------------------------
    # Workflow graph construction
    # ------------------------------------------------------------------

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph StateGraph (v3).

        Nodes (8):  decomposer, planner, retrieval, synthesis,
                    reranker, validator, writer, critic

        Fixed edges (7):
            decomposer → planner
            planner    → retrieval
            retrieval  → synthesis
            synthesis  → reranker
            reranker   → validator
            writer     → critic
            (validator → writer via conditional PROCEED)

        Conditional edges (2):
            validator  → retrieval (retry) | writer (proceed)
            critic     → writer (regenerate) | END (finish)
        """
        self.logger.info("Building LangGraph workflow (v3)…")

        class WorkflowState(TypedDict):
            agent_state: AgentState

        graph = StateGraph(WorkflowState)

        # ── Nodes ──────────────────────────────────────────────────────
        graph.add_node("decomposer", self._decomposer_node_wrapper)
        graph.add_node("planner",    self._planner_node_wrapper)
        graph.add_node("retrieval",  self._retrieval_node_wrapper)
        graph.add_node("synthesis",  self._synthesis_node_wrapper)
        graph.add_node("reranker",   self._reranker_node_wrapper)
        graph.add_node("validator",  self._validator_node_wrapper)
        graph.add_node("writer",     self._writer_node_wrapper)
        graph.add_node("critic",     self._critic_node_wrapper)

        # ── Stage 1: Decomposition → Planning ─────────────────────────
        graph.set_entry_point("decomposer")
        graph.add_edge("decomposer", "planner")
        graph.add_edge("planner",    "retrieval")

        # ── Stage 2: Retrieval → Synthesis → Reranker → Validator ─────
        graph.add_edge("retrieval", "synthesis")
        graph.add_edge("synthesis", "reranker")
        graph.add_edge("reranker",  "validator")

        # Validator: retry goes back to retrieval (→ synthesis → reranker
        # → validator again via fixed edges — no extra wiring needed).
        graph.add_conditional_edges(
            "validator",
            self._should_retry_retrieval_wrapper,
            {
                "retry":   "retrieval",
                "proceed": "writer",
            },
        )

        # ── Stage 3: Generation with self-reflection ───────────────────
        graph.add_edge("writer", "critic")
        graph.add_conditional_edges(
            "critic",
            self._should_regenerate_wrapper,
            {
                "regenerate": "writer",
                "finish":     END,
            },
        )

        compiled = graph.compile()
        self.logger.info(
            "Workflow built: 8 nodes, 6 fixed edges, 2 conditional edges"
        )
        return compiled

    # ------------------------------------------------------------------
    # Node execution methods
    # ------------------------------------------------------------------

    def _decomposer_node(self, state: AgentState) -> AgentState:
        self.logger.info("DECOMPOSER node — LLM decomposition decision")
        try:
            result = self.decomposer.run(state)
            meta = result.metadata.get("decomposition", {})
            self.logger.info(
                format_stage_trace(
                    "decomposer",
                    inputs={"query": state.query[:80]},
                    outputs={
                        "need_decompose": meta.get("need_decompose", "?"),
                        "sub_queries": len(result.sub_queries or []),
                    },
                    metrics={
                        "reason":   meta.get("decision_reason", ""),
                        "queries":  result.sub_queries or [],
                    },
                )
            )
            self.logger.info(
                f"Decomposer: need_decompose={meta.get('need_decompose', '?')}, "
                f"{len(result.sub_queries or [])} sub-queries"
            )
            return result
        except Exception as exc:
            self.logger.error(f"Decomposer node failed: {exc}")
            raise OrchestrationError(
                node_name="decomposer",
                message=f"Decomposer failed: {exc}",
                details={"query": state.query},
            ) from exc

    def _planner_node(self, state: AgentState) -> AgentState:
        self.logger.info("PLANNER node — per-sub-query retrieval strategy")
        try:
            result = self.planner.run(state)
            meta = result.metadata.get("planner", {})
            plans = result.sub_query_plans or []
            self.logger.info(
                format_stage_trace(
                    "planner",
                    inputs={"sub_queries": len(state.sub_queries or [])},
                    outputs={"plans": len(plans)},
                    metrics={
                        "complexity": (
                            f"{result.complexity:.2f}"
                            if result.complexity is not None
                            else "n/a"
                        ),
                    },
                )
            )
            for i, plan in enumerate(plans):
                q_preview = plan.get("query", "")[:70]
                self.logger.info(
                    f"  [{i + 1}/{len(plans)}] retrievers={plan.get('retrievers')} "
                    f'query="{q_preview}"'
                )
            return result
        except Exception as exc:
            self.logger.error(f"Planner node failed: {exc}")
            raise OrchestrationError(
                node_name="planner",
                message=f"Planner failed: {exc}",
                details={"query": state.query},
            ) from exc

    def _retrieval_node(self, state: AgentState) -> AgentState:
        n_plans = len(state.sub_query_plans or [])
        if n_plans:
            self.logger.info(
                f"RETRIEVAL node (round {state.retrieval_round}) — "
                f"{n_plans} sub-query plan(s)"
            )
        else:
            self.logger.info(
                f"RETRIEVAL node (round {state.retrieval_round}) — "
                f"retrievers={state.selected_retrievers}"
            )
        try:
            result = self.coordinator.run(state)
            meta = result.metadata.get("retrieval_coordinator", {})
            self.logger.info(
                format_stage_trace(
                    "retrieval",
                    inputs={
                        "retrievers": meta.get("selected_retrievers", []),
                        "requested": meta.get("retriever_quotas", {}),
                    },
                    outputs={
                        "raw_chunks": meta.get("total_retrieved", 0),
                        "unique_chunks": meta.get("unique_chunks", len(result.chunks)),
                    },
                    metrics={
                        "round": f"{meta.get('round', 0)}->{result.retrieval_round}",
                        "path": meta.get("path", "unknown"),
                        "breakdown": meta.get("retriever_results", {}),
                        "chunks": summarize_chunks(result.chunks),
                    },
                )
            )
            self.logger.info(
                f"Retrieval: {len(result.chunks)} chunks "
                f"(round {result.retrieval_round})"
            )
            return result
        except Exception as exc:
            self.logger.error(f"Retrieval node failed: {exc}")
            raise OrchestrationError(
                node_name="retrieval",
                message=f"Retrieval failed: {exc}",
                details={"query": state.query, "round": state.retrieval_round},
            ) from exc

    def _synthesis_node(self, state: AgentState) -> AgentState:
        self.logger.info("SYNTHESIS node — dedup + candidate pool")
        try:
            result = self.synthesis.run(state)
            meta   = result.metadata.get("synthesis", {})
            self.logger.info(
                format_stage_trace(
                    "synthesis",
                    inputs={"raw_chunks": meta.get("input_count", 0)},
                    outputs={"candidate_chunks": meta.get("unique_count", 0)},
                    metrics={
                        "dedup_rate": meta.get("deduplication_rate", 0),
                        "sources": meta.get("source_breakdown", {}),
                    },
                )
            )
            self.logger.info(
                f"Synthesis: {meta.get('input_count', 0)} → "
                f"{meta.get('unique_count', 0)} unique chunks "
                f"(sources: {meta.get('source_breakdown', {})})"
            )
            return result
        except Exception as exc:
            self.logger.error(f"Synthesis node failed: {exc}")
            raise OrchestrationError(
                node_name="synthesis",
                message=f"Synthesis failed: {exc}",
                details={"query": state.query},
            ) from exc

    def _reranker_node(self, state: AgentState) -> AgentState:
        self.logger.info("RERANKER node — Cohere rerank / weighted fallback")
        try:
            result = self.reranker.run(state)
            meta   = result.metadata.get("reranker", {})
            self.logger.info(
                format_stage_trace(
                    "reranker",
                    inputs={"candidate_chunks": meta.get("input_count", 0)},
                    outputs={"final_chunks_for_llm": meta.get("final_count", 0)},
                    metrics={
                        "top_k": meta.get("top_k", "n/a"),
                        "cohere": meta.get("used_cohere", False),
                        "model": meta.get("model", "n/a"),
                        "chunks": summarize_chunks(result.chunks),
                    },
                )
            )
            self.logger.info(
                f"Reranker: {meta.get('input_count', 0)} → "
                f"{meta.get('final_count', 0)} chunks "
                f"(cohere={meta.get('used_cohere', False)})"
            )
            return result
        except Exception as exc:
            self.logger.error(f"Reranker node failed: {exc}")
            raise OrchestrationError(
                node_name="reranker",
                message=f"Reranker failed: {exc}",
                details={"query": state.query},
            ) from exc

    def _validator_node(self, state: AgentState) -> AgentState:
        self.logger.info("VALIDATOR node")
        try:
            result = self.validator.run(state)
            self.logger.info(
                format_stage_trace(
                    "validator",
                    inputs={"chunks": len(state.chunks)},
                    outputs={"decision": result.validation_status},
                    metrics={
                        "score": (
                            f"{result.validation_score:.3f}"
                            if result.validation_score is not None
                            else "n/a"
                        ),
                        "threshold": result.metadata.get("validator", {}).get("threshold", "n/a"),
                        "round": (
                            f"{state.retrieval_round}/"
                            f"{result.metadata.get('validator', {}).get('max_retries', 'n/a')}"
                        ),
                    },
                )
            )
            self.logger.info(
                f"Validator: score={result.validation_score:.2f}, "
                f"decision={result.validation_status}"
            )
            return result
        except Exception as exc:
            self.logger.error(f"Validator node failed: {exc}")
            raise OrchestrationError(
                node_name="validator",
                message=f"Validator failed: {exc}",
                details={"query": state.query},
            ) from exc

    def _writer_node(self, state: AgentState) -> AgentState:
        regeneration_count = state.metadata.get("regeneration_count", 0)

        if regeneration_count > 0:
            self.logger.info(
                f"WRITER node (regeneration {regeneration_count})"
            )
            try:
                improved = self.writer.generate_with_feedback(
                    query=state.query,
                    chunks=state.chunks,
                    feedback=state.critic_feedback,
                )
                state.answer = improved
                state.metadata["regeneration_count"] = regeneration_count
                self.logger.info(
                    f"Writer: regenerated ({len(improved)} chars)"
                )
                return state
            except Exception as exc:
                self.logger.error(f"Writer regeneration failed: {exc}")
                raise OrchestrationError(
                    node_name="writer_regenerate",
                    message=f"Writer regeneration failed: {exc}",
                    details={"query": state.query},
                ) from exc
        else:
            self.logger.info("WRITER node (initial generation)")
            try:
                result = self.writer.run(state)
                meta = result.metadata.get("writer", {})
                self.logger.info(
                    format_stage_trace(
                        "writer",
                        inputs={"chunks_for_llm": meta.get("chunks_used", len(state.chunks))},
                        outputs={
                            "answer_chars": meta.get("answer_length", len(result.answer or "")),
                            "llm_cited_chunks": meta.get("citations_count", 0),
                        },
                        metrics={
                            "citation_ids": meta.get("citation_ids", []),
                            "citation_child_ids": meta.get("citation_child_ids", []),
                        },
                    )
                )
                self.logger.info(
                    f"Writer: {len(result.answer)} chars generated"
                )
                return result
            except Exception as exc:
                self.logger.error(f"Writer node failed: {exc}")
                raise OrchestrationError(
                    node_name="writer",
                    message=f"Writer failed: {exc}",
                    details={"query": state.query},
                ) from exc

    def _critic_node(self, state: AgentState) -> AgentState:
        self.logger.info("CRITIC node")
        try:
            result = self.critic.run(state)
            self.logger.info(
                format_stage_trace(
                    "critic",
                    inputs={"answer_chars": len(state.answer or "")},
                    outputs={"decision": result.critic_decision.value},
                    metrics={
                        "score": (
                            f"{result.critic_score:.3f}"
                            if result.critic_score is not None
                            else "n/a"
                        ),
                        "regenerations": result.metadata.get("regeneration_count", 0),
                    },
                )
            )
            self.logger.info(
                f"Critic: score={result.critic_score:.2f}, "
                f"decision={result.critic_decision.value}"
            )
            return result
        except Exception as exc:
            self.logger.error(f"Critic node failed: {exc}")
            raise OrchestrationError(
                node_name="critic",
                message=f"Critic failed: {exc}",
                details={"query": state.query},
            ) from exc

    # ------------------------------------------------------------------
    # Conditional routing
    # ------------------------------------------------------------------

    def _should_retry_retrieval(
        self, state: AgentState
    ) -> Literal["retry", "proceed"]:
        decision = state.validation_status
        if decision == "PROCEED":
            self.logger.info("Validator PASSED → proceed to writer")
            return "proceed"
        if decision == "RETRIEVE_MORE":
            self._expand_retrievers_for_retry(state)
            self.logger.info(
                f"Validator FAILED → retry retrieval "
                f"(round {state.retrieval_round}, "
                f"retrievers={state.selected_retrievers})"
            )
            return "retry"
        # Unknown status — default to proceed
        self.logger.warning(
            f"Unknown validation status '{decision}' → proceeding"
        )
        return "proceed"

    def _expand_retrievers_for_retry(self, state: AgentState) -> None:
        """On validator retry, force the next retrieval round to use all paths."""
        target_retrievers = ["vector", "keyword", "graph"]
        before = list(state.selected_retrievers or [])
        existing_quotas = dict(state.retriever_quotas or {})
        default_quota = 10

        state.selected_retrievers = target_retrievers
        state.retriever_quotas = {
            name: existing_quotas.get(name, default_quota)
            for name in target_retrievers
        }
        state.metadata["retry_retrieval_expansion"] = {
            "round": state.retrieval_round,
            "before_retrievers": before,
            "after_retrievers": target_retrievers,
            "reason": "validator_retrieve_more",
        }

        self.logger.info(
            f"Retry retrieval expansion: {before} → {target_retrievers}, "
            f"quotas={state.retriever_quotas}"
        )

    def _should_regenerate(
        self, state: AgentState
    ) -> Literal["regenerate", "finish"]:
        decision           = state.critic_decision
        regeneration_count = state.metadata.get("regeneration_count", 0)
        max_iterations     = self.critic.max_iterations

        if decision == CriticDecision.APPROVED:
            self.logger.info("Critic APPROVED → finish")
            return "finish"

        if decision == CriticDecision.REGENERATE:
            if regeneration_count < max_iterations:
                self.logger.info(
                    f"Critic REGENERATE "
                    f"({regeneration_count + 1}/{max_iterations})"
                )
                state.metadata["regeneration_count"] = regeneration_count + 1
                return "regenerate"
            self.logger.warning(
                f"Max iterations reached ({max_iterations}) → finish"
            )
            return "finish"

        self.logger.info(f"Critic decision: {decision.value} → finish")
        return "finish"

    # ------------------------------------------------------------------
    # LangGraph node/edge wrappers
    # ------------------------------------------------------------------

    def _decomposer_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._decomposer_node(state["agent_state"])}

    def _planner_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._planner_node(state["agent_state"])}

    def _retrieval_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._retrieval_node(state["agent_state"])}

    def _synthesis_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._synthesis_node(state["agent_state"])}

    def _reranker_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._reranker_node(state["agent_state"])}

    def _validator_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._validator_node(state["agent_state"])}

    def _writer_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._writer_node(state["agent_state"])}

    def _critic_node_wrapper(self, state: dict) -> dict:
        return {"agent_state": self._critic_node(state["agent_state"])}

    def _should_retry_retrieval_wrapper(
        self, state: dict
    ) -> Literal["retry", "proceed"]:
        return self._should_retry_retrieval(state["agent_state"])

    def _should_regenerate_wrapper(
        self, state: dict
    ) -> Literal["regenerate", "finish"]:
        return self._should_regenerate(state["agent_state"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> AgentState:
        """
        Execute the full pipeline for a user query.

        Parameters
        ----------
        query : str

        Returns
        -------
        AgentState — final state with answer, citations, and all metadata.
        """
        self.logger.info(f"Starting workflow for: {query[:80]}…")
        try:
            initial_state = {"agent_state": AgentState(query=query)}
            final_state   = self.workflow.invoke(initial_state)
            result        = final_state["agent_state"]

            self.logger.info(
                f"Workflow complete — "
                f"retrievers={result.selected_retrievers}, "
                f"chunks={len(result.chunks)}, "
                f"rounds={result.retrieval_round}, "
                f"critic_score={result.critic_score:.2f}, "
                f"regenerations={result.metadata.get('regeneration_count', 0)}"
            )
            return result

        except OrchestrationError:
            raise
        except Exception as exc:
            self.logger.error(f"Workflow failed: {exc}")
            raise OrchestrationError(
                node_name="workflow",
                message=f"Workflow execution failed: {exc}",
                details={"query": query},
            ) from exc

    def run_with_trace(self, query: str) -> Dict[str, Any]:
        """
        Run the workflow and return a detailed execution trace.

        Useful for debugging, evaluation, and UI display.

        Returns
        -------
        dict with keys:
            final_state, execution_path, node_timings, total_duration,
            retrieval_attempts, regeneration_attempts
        """
        import time

        self.logger.info(f"Starting traced workflow for: {query[:80]}…")

        execution_path: list  = []
        node_timings:   dict  = {}
        state = AgentState(query=query)

        def timed(name: str, fn):
            t0 = time.time()
            execution_path.append(name)
            result = fn()
            elapsed = time.time() - t0
            if name not in node_timings:
                node_timings[name] = []
            node_timings[name].append(elapsed)
            return result

        try:
            # Stage 1: Decomposition + Planning
            state = timed("decomposer", lambda: self._decomposer_node(state))
            state = timed("planner",    lambda: self._planner_node(state))

            # Stage 2: Retrieval loop
            retrieval_attempts = 0
            max_attempts       = 5
            while retrieval_attempts < max_attempts:
                state = timed("retrieval", lambda: self._retrieval_node(state))
                state = timed("synthesis", lambda: self._synthesis_node(state))
                state = timed("reranker",  lambda: self._reranker_node(state))
                state = timed("validator", lambda: self._validator_node(state))
                if self._should_retry_retrieval(state) == "proceed":
                    break
                retrieval_attempts += 1

            # Stage 3: Generation loop
            regeneration_attempts = 0
            max_regen             = self.critic.max_iterations
            while regeneration_attempts <= max_regen:
                state = timed("writer", lambda: self._writer_node(state))
                state = timed("critic", lambda: self._critic_node(state))
                if self._should_regenerate(state) == "finish":
                    break
                regeneration_attempts += 1

            return {
                "final_state":           state,
                "execution_path":        execution_path,
                "node_timings":          node_timings,
                "total_duration":        sum(
                    t for ts in node_timings.values() for t in ts
                ),
                "retrieval_attempts":    retrieval_attempts + 1,
                "regeneration_attempts": regeneration_attempts,
            }

        except Exception as exc:
            self.logger.error(f"Traced workflow failed: {exc}")
            raise OrchestrationError(
                node_name="workflow_trace",
                message=f"Traced execution failed: {exc}",
                details={"query": query, "execution_path": execution_path},
            ) from exc

    def get_workflow_info(self) -> Dict[str, Any]:
        """Return static metadata about the workflow structure."""
        return {
            "version": "3.0.0",
            "nodes": [
                "decomposer", "planner", "retrieval",
                "synthesis", "reranker", "validator",
                "writer", "critic",
            ],
            "edges": {
                "fixed": [
                    "START → decomposer",
                    "decomposer → planner",
                    "planner → retrieval",
                    "retrieval → synthesis",
                    "synthesis → reranker",
                    "reranker → validator",
                    "writer → critic",
                ],
                "conditional": [
                    "validator  → retrieval (retry) | writer (proceed)",
                    "critic     → writer (regenerate) | END (finish)",
                ],
            },
            "retry_mechanisms": {
                "retrieval":  "validator triggers retry → retrieval → synthesis → reranker",
                "generation": "critic triggers regeneration → writer",
            },
            "max_retries": {
                "retrieval":  self.validator.max_retries,
                "generation": self.critic.max_iterations,
            },
            "reranker": {
                "model":    self.reranker.cohere_model,
                "top_k":    self.reranker.top_k,
                "fallback": "weighted-score (vector:0.7, keyword:0.3, graph:0.9)",
            },
        }
