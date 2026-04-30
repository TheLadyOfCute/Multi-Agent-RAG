"""
Reranker Agent - Standalone reranking node in the LangGraph workflow.

Sits between RetrievalCoordinator and ValidatorAgent:

    Retrieval -> Reranker -> Validator

Responsibilities
----------------
1. Call Cohere ``rerank-english-v3.0`` on the full candidate pool
   produced by RetrievalCoordinator.
2. Apply top-k truncation after reranking.
3. Fall back to a source-weighted scoring strategy when the Cohere
   API is unavailable or raises an error — preserving a deterministic,
   reasonable ordering without blocking the pipeline.

Fallback weight scheme
----------------------

    source     weight
    ------     ------
    vector     0.70   (semantic relevance)
    keyword    0.30   (exact match, already filtered to relevant docs)
    graph      0.90   (traversal relevance, high precision)
    <unknown>  0.50   (neutral)
"""

from typing import List, Dict

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.config import get_settings
from src.utils.retrieval_debug import format_ranked_chunk_line


# Weights used in the fallback ranking strategy
_FALLBACK_WEIGHTS: Dict[str, float] = {
    "vector":  0.70,
    "keyword": 0.30,
    "graph":   0.90,
}
_FALLBACK_DEFAULT_WEIGHT = 0.50


class RerankerAgent(BaseAgent):

    def __init__(
        self,
        top_k: int = None,
        cohere_model: str = "rerank-v4.0-pro",
    ):
        super().__init__(name="reranker", version="1.0.0")
        settings        = get_settings()
        self.top_k      = top_k or settings.retrieval_top_k
        self.cohere_model = cohere_model
        self.log(
            f"Initialized: top_k={self.top_k}, model={self.cohere_model}",
            level="info",
        )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> AgentState:

        chunks = state.chunks

        if not chunks:
            self.log("No chunks to rerank", level="warning")
            return state

        input_count = len(chunks)
        self.log(f"Reranking {input_count} candidate chunks", level="info")

        # ── Attempt Cohere reranking ───────────────────────────────────
        reranked, used_cohere = self._rerank_with_cohere(state.query, chunks)

        if not used_cohere:
            self.log(
                "Cohere unavailable — using weighted-score fallback",
                level="warning",
            )
            reranked = self._fallback_weight_rank(chunks)

        # ── Top-k truncation ──────────────────────────────────────────
        final = reranked[: self.top_k]

        self.log(
            f"Reranking complete: {input_count} → {len(final)} chunks "
            f"(cohere={'yes' if used_cohere else 'fallback'})",
            level="info",
        )
        print(
            f"\nReranker Results "
            f"(input={input_count}, final={len(final)}, "
            f"mode={'cohere' if used_cohere else 'fallback'}, top_k={self.top_k})"
        )
        for rank, chunk in enumerate(final, start=1):
            print(format_ranked_chunk_line(rank, chunk))
        
        state.chunks = final
        state.metadata["reranker"] = {
            "input_count":  input_count,
            "final_count":  len(final),
            "top_k":        self.top_k,
            "model":        self.cohere_model if used_cohere else "fallback-weights",
            "used_cohere":  used_cohere,
        }

        return state

    # ------------------------------------------------------------------
    # Cohere reranking
    # ------------------------------------------------------------------

    def _rerank_with_cohere(
        self, query: str, chunks: List[Chunk]
    ) -> tuple[List[Chunk], bool]:
        
        try:
            import cohere  # noqa: PLC0415
        except ImportError:
            self.log(
                "cohere package not installed — run: pip install cohere",
                level="warning",
            )
            return chunks, False

        settings = get_settings()
        if not settings.cohere_api_key:
            self.log("COHERE_API_KEY not set", level="warning")
            return chunks, False

        try:
            co=cohere.Client(api_key=settings.cohere_api_key)
            documents = [c.text for c in chunks]

            self.log(
                f"Calling Cohere {self.cohere_model} with "
                f"{len(documents)} documents…",
                level="debug",
            )

            response = co.rerank(
                model=self.cohere_model,
                query=query,
                documents=documents,
                top_n=len(chunks),      # rerank all; we truncate to top_k later
                return_documents=False,
            )

            # Reorder chunks according to Cohere ranking
            reranked: List[Chunk] = []
            for result in response.results:
                chunk = chunks[result.index]
                chunk.metadata["rerank_score"]= result.relevance_score
                chunk.metadata["pre_rerank_score"]= chunk.score
                chunk.score=result.relevance_score
                reranked.append(chunk)

            return reranked, True

        except Exception as exc:
            self.log(f"Cohere reranking failed: {exc}", level="warning")
            return chunks, False

    # ------------------------------------------------------------------
    # Weighted-score fallback
    # ------------------------------------------------------------------

    def _fallback_weight_rank(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rank chunks by source-weighted score when Cohere is unavailable.

        Each chunk's score is multiplied by a weight determined by
        which retriever produced it (vector / keyword / graph).
        Chunks are then sorted descending by the weighted score.

        Parameters
        ----------
        chunks : list of Chunk

        Returns
        -------
        list of Chunk, sorted by weighted score descending.
        """
        ranked = []
        for chunk in chunks:
            source = (
                chunk.metadata.get("retriever")
                or chunk.metadata.get("source")
                or "unknown"
            )
            # 处理合并来源 "vector|keyword|graph"：取最大权重
            if "|" in source:
                w = max(
                    _FALLBACK_WEIGHTS.get(s.strip(), _FALLBACK_DEFAULT_WEIGHT)
                    for s in source.split("|")
                )
            else:
                w = _FALLBACK_WEIGHTS.get(source, _FALLBACK_DEFAULT_WEIGHT)
            base_score = chunk.score or 0.0
            weighted   = base_score * w

            chunk.metadata["fallback_weight"]        = w
            chunk.metadata["fallback_weighted_score"] = weighted
            chunk.metadata["pre_rerank_score"]        = base_score
            chunk.score                               = weighted

            ranked.append(chunk)

        return sorted(ranked, key=lambda c: c.score, reverse=True)
