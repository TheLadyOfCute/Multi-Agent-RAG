from typing import Dict, Any, Literal

try:
    from typing import TypeAlias
except ImportError:
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        TypeAlias = Any

from langgraph.graph import StateGraph, END

from src.models.agent_state import AgentState
from src.agents.planner import PlannerAgent
from src.agents.query_decomposer import QueryDecomposer
from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.validator import ValidatorAgent
from src.agents.reranker import RerankerAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent, CriticDecision
from src.utils.logger import setup_logger
from src.utils.workflow_trace import format_stage_trace, summarize_chunks
from src.utils.exceptions import OrchestrationError


WorkflowState: TypeAlias = Dict[str, Any]


class CompleteAgenticRAGWorkflow:

    def __init__(
        self,
        planner:     PlannerAgent,
        decomposer:  QueryDecomposer,
        coordinator: RetrievalCoordinator,
        validator:   ValidatorAgent,
        reranker:    RerankerAgent,
        writer:      WriterAgent,
        critic:      CriticAgent,
        vector_store=None,
    ):
        self.planner     = planner
        self.decomposer  = decomposer
        self.coordinator = coordinator
        self.validator   = validator
        self.reranker    = reranker
        self.writer      = writer
        self.critic      = critic
        self.vector_store = vector_store

        self.logger = setup_logger("complete_workflow", level="INFO")
        self.workflow = self._build_workflow()
        self.logger.info(
            "Complete AgenticRAG workflow v3 initialized (7 nodes)"
        )

    # ------------------------------------------------------------------
    # Workflow graph construction
    # ------------------------------------------------------------------

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph StateGraph (v3).

        Nodes (7):  decomposer, planner, retrieval,
                    reranker, validator, writer, critic

        Fixed edges (5):
            decomposer -> planner
            planner    -> retrieval
            retrieval  -> reranker
            reranker   -> validator
            writer     -> critic

        Conditional edges (2):
            validator -> retrieval (retry) | writer (proceed)
            critic    -> writer (regenerate) | END (finish)
        """
        self.logger.info("Building LangGraph workflow (v3)")

        graph = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────────────
        graph.add_node("decomposer", self._decomposer_node)
        graph.add_node("planner",    self._planner_node)
        graph.add_node("retrieval",  self._retrieval_node)
        graph.add_node("reranker",   self._reranker_node)
        graph.add_node("validator",  self._validator_node)
        graph.add_node("writer",     self._writer_node)
        graph.add_node("critic",     self._critic_node)

        # ── Stage 1: Decomposition + Planning ──────────────────────────
        graph.set_entry_point("decomposer")
        graph.add_edge("decomposer", "planner")
        graph.add_edge("planner",    "retrieval")

        # ── Stage 2: Retrieval -> Reranker -> Validator ────────────────
        graph.add_edge("retrieval", "reranker")
        graph.add_edge("reranker",  "validator")

        # Validator: retry goes back to retrieval
        graph.add_conditional_edges(
            "validator",
            self._should_retry_retrieval,
            {
                "retry":   "retrieval",
                "proceed": "writer",
            },
        )

        # ── Stage 3: Generation with self-reflection ───────────────────
        graph.add_edge("writer", "critic")
        graph.add_conditional_edges(
            "critic",
            self._should_regenerate,
            {
                "regenerate": "writer",
                "finish":     END,
            },
        )

        compiled = graph.compile()
        self.logger.info(
            "Workflow built: 7 nodes, 5 fixed edges, 2 conditional edges"
        )
        return compiled

    # ------------------------------------------------------------------
    # Node execution methods
    # ------------------------------------------------------------------

    def _decomposer_node(self, state: AgentState) -> AgentState:
        self.logger.info("DECOMPOSER nodeLLM decomposition decision")
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
        self.logger.info("PLANNER nodeper-sub-query retrieval strategy")
        try:
            result = self.planner.run(state)
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
                f"RETRIEVAL node (round {state.retrieval_round})"
                f"{n_plans} sub-query plan(s)"
            )
        else:
            self.logger.info(
                f"RETRIEVAL node (round {state.retrieval_round})"
            )
        try:
            result = self.coordinator.run(state)
            meta = result.metadata.get("retrieval_coordinator", {})
            self.logger.info(
                format_stage_trace(
                    "retrieval",
                    inputs={
                        "plans": meta.get("sub_query_plans", 0),
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

    def _reranker_node(self, state: AgentState) -> AgentState:
        self.logger.info("RERANKER nodeCohere rerank / weighted fallback")
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
                f"Reranker: {meta.get('input_count', 0)}"
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
                            "citation_chunk_ids": meta.get("citation_chunk_ids", []),
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
            self.logger.info("Validator PASSEDproceed to writer")
            return "proceed"
        if decision == "RETRIEVE_MORE":
            self._expand_retrievers_for_retry(state)
            self.logger.info(
                f"Validator FAILEDretry retrieval "
                f"(round {state.retrieval_round})"
            )
            return "retry"
        # Unknown statusdefault to proceed
        self.logger.warning(
            f"Unknown validation status '{decision}'proceeding"
        )
        return "proceed"

    def _expand_retrievers_for_retry(self, state: AgentState) -> None:
        """On validator retry, force the next retrieval round to use all paths."""
        target_retrievers = ["vector", "keyword", "graph"]
        default_quota = 10

        before_plans = []
        for plan in state.sub_query_plans or []:
            plan_quotas = dict(plan.get("quotas") or {})
            before_plans.append(
                {
                    "query": plan.get("query", state.query),
                    "retrievers": list(plan.get("retrievers") or []),
                    "quotas": plan_quotas,
                }
            )
            plan["retrievers"] = list(target_retrievers)
            plan["quotas"] = {
                name: plan_quotas.get(name, default_quota)
                for name in target_retrievers
            }

        state.metadata["retry_retrieval_expansion"] = {
            "round": state.retrieval_round,
            "before_sub_query_plans": before_plans,
            "after_sub_query_plans": state.sub_query_plans or [],
            "reason": "validator_retrieve_more",
        }

        self.logger.info(
            f"Retry retrieval expansion: all retrievers forced, "
            f"plans={len(state.sub_query_plans or [])}"
        )

    def _should_regenerate(
        self, state: AgentState
    ) -> Literal["regenerate", "finish"]:
        decision= state.critic_decision
        regeneration_count = state.metadata.get("regeneration_count", 0)
        max_iterations= self.critic.max_iterations

        if decision == CriticDecision.APPROVED:
            self.logger.info("Critic APPROVEDfinish")
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
                f"Max iterations reached ({max_iterations})finish"
            )
            return "finish"

        self.logger.info(f"Critic decision: {decision.value}finish")
        return "finish"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> WorkflowState:
        self.logger.info(f"Starting workflow for: {query[:80]}")
        try:
            initial_state = AgentState(query=query)

            # 设置知识库总文档数，供 validator 评估多样性
            vector_store = getattr(self, "vector_store", None)
            if vector_store and hasattr(vector_store, "count_documents"):
                initial_state.total_docs = vector_store.count_documents()

            raw_result = self.workflow.invoke(initial_state)
            result = self._normalize_workflow_result(raw_result)

            self.logger.info(
                f"Workflow complete"
                f"chunks={len(result.get('chunks') or [])}, "
                f"rounds={result.get('retrieval_round')}, "
                f"critic_score={self._format_optional_score(result.get('critic_score'))}, "
                f"regenerations={result.get('metadata', {}).get('regeneration_count', 0)}"
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

    @staticmethod
    def _normalize_workflow_result(result: Any) -> WorkflowState:
        if isinstance(result, dict):
            return dict(result)

        if isinstance(result, AgentState):
            return {
                "query": result.query,
                "complexity": result.complexity,
                "strategy": result.strategy,
                "sub_queries": result.sub_queries,
                "sub_query_plans": result.sub_query_plans,
                "chunks": result.chunks,
                "retrieval_round": result.retrieval_round,
                "validation_status": result.validation_status,
                "validation_score": result.validation_score,
                "answer": result.answer,
                "critic_score": result.critic_score,
                "critic_feedback": result.critic_feedback,
                "critic_scores": result.critic_scores,
                "critic_decision": result.critic_decision,
                "metadata": result.metadata,
            }

        raise TypeError(f"Unsupported workflow result type: {type(result).__name__}")

    @staticmethod
    def _format_optional_score(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

