"""
Synthesis Agent - Candidate Pool Builder.

Responsibility (v2):
  1. Deduplicate chunks arriving from multiple retrievers (MD5 hash).
  2. Preserve source / retriever metadata on every chunk.
  3. Return the full unique candidate pool to the downstream
     RerankerAgent — no hybrid-rank scoring, no top-k truncation here.

The old _hybrid_rank and _rerank_with_cohere logic has been moved to
RerankerAgent (src/agents/reranker.py) which sits as a dedicated
LangGraph node between Synthesis and Validator.
"""

from typing import List, Dict, Any
import hashlib
from collections import defaultdict

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.config import get_settings
from src.utils.retrieval_debug import chunk_dedup_key, merge_retriever_sources


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent — Deduplication and candidate pool construction.

    Receives all chunks from the RetrievalCoordinator (potentially
    produced by multiple retrievers: vector, keyword, graph) and
    returns a deduplicated candidate pool with source information
    intact.

    Ranking / reranking is intentionally NOT performed here; that is
    the responsibility of the separate RerankerAgent that follows this
    node in the LangGraph workflow.

    Attributes
    ----------
    use_reranker : bool
        Deprecated parameter kept for backward compatibility.
        Has no effect — reranking is now done by RerankerAgent.

    Example
    -------
    >>> agent = SynthesisAgent()
    >>> state = AgentState(query="query", chunks=[...])
    >>> result = agent.run(state)
    >>> # result.chunks contains deduplicated pool (no top-k truncation)
    """

    def __init__(
        self,
        top_k: int = None,             # kept for compat; unused
        vector_weight: float = None,   # kept for compat; unused
        keyword_weight: float = None,  # kept for compat; unused
        use_reranker: bool = False,    # deprecated; has no effect
    ):
        super().__init__(name="synthesis", version="2.0.0")

        if use_reranker:
            self.log(
                "use_reranker=True is deprecated on SynthesisAgent. "
                "Reranking is now handled by the standalone RerankerAgent node.",
                level="warning",
            )

        settings = get_settings()
        # Store top_k only for metadata reporting; actual truncation
        # happens in RerankerAgent.
        self._config_top_k = top_k or settings.retrieval_top_k

        self.log(
            f"Initialized (v2: dedup + candidate pool only, "
            f"config_top_k={self._config_top_k})",
            level="info",
        )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> AgentState:
        """
        Build deduplicated candidate pool from multi-retriever chunks.

        Steps
        -----
        1. Deduplicate by content hash (keep highest-scored copy).
        2. Preserve ``source`` / ``retriever`` metadata on each chunk.
        3. Pass the full unique pool downstream (no top-k cut here).

        Parameters
        ----------
        state : AgentState
            Must carry ``state.chunks`` populated by
            RetrievalCoordinator.

        Returns
        -------
        AgentState
            ``state.chunks`` replaced with the deduplicated candidate
            pool.
        """
        try:
            chunks = state.chunks

            if not chunks:
                self.log("No chunks to synthesize", level="warning")
                return state

            self.log(
                f"Building candidate pool from {len(chunks)} raw chunks",
                level="info",
            )

            # ── Step 1: Deduplicate ────────────────────────────────────
            unique_chunks = self._deduplicate(chunks)
            self.log(
                f"Deduplication: {len(chunks)} → {len(unique_chunks)} unique chunks",
                level="debug",
            )

            # ── Step 2: Collect source breakdown for metadata ──────────
            source_counts: Dict[str, int] = defaultdict(int)
            for c in unique_chunks:
                src = c.metadata.get("retriever") or c.metadata.get("source", "unknown")
                source_counts[src] += 1

            # ── Step 3: Write candidate pool (full, not truncated) ─────
            state.chunks = unique_chunks

            dedup_rate = (
                1.0 - len(unique_chunks) / len(chunks) if chunks else 0.0
            )

            state.metadata["synthesis"] = {
                "input_count":       len(chunks),
                "unique_count":      len(unique_chunks),
                "deduplication_rate": round(dedup_rate, 4),
                "source_breakdown":  dict(source_counts),
                "top_k_config":      self._config_top_k,
                "note": (
                    "Candidate pool passed to RerankerAgent for "
                    "Cohere reranking + top-k selection."
                ),
            }

            self.log(
                f"Candidate pool ready: {len(unique_chunks)} chunks "
                f"(sources: {dict(source_counts)})",
                level="info",
            )

            return state

        except Exception as exc:
            self.log(f"Synthesis failed: {exc}", level="error")
            return state

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove duplicate chunks using MD5 content hash.

        When multiple copies of the same text exist (e.g. returned by
        both vector and keyword retrievers) only the copy with the
        highest score is kept.  All other metadata (source tags) from
        duplicate copies is merged into the kept chunk's metadata so
        traceability is not lost.

        Parameters
        ----------
        chunks : list of Chunk

        Returns
        -------
        list of Chunk — unique chunks sorted by original score desc.
        """
        if not chunks:
            return []

        hash_groups: Dict[str, List[Chunk]] = defaultdict(list)
        for chunk in chunks:
            h = chunk_dedup_key(chunk)
            hash_groups[h].append(chunk)

        unique: List[Chunk] = []
        for group in hash_groups.values():
            sorted_group = sorted(
                group,
                key=lambda c: c.score or 0.0,
                reverse=True,
            )
            best = sorted_group[0]
            merge_retriever_sources(best, sorted_group[1:])

            # Merge retriever/source tags from duplicates
            if len(sorted_group) > 1:
                extra_sources = {
                    c.metadata.get("retriever") or c.metadata.get("source")
                    for c in sorted_group[1:]
                    if c.metadata.get("retriever") or c.metadata.get("source")
                }
                if extra_sources:
                    existing = best.metadata.get("also_from", [])
                    best.metadata["also_from"] = list(
                        set(existing) | extra_sources
                    )

            unique.append(best)

        return unique

    def _compute_hash(self, text: str) -> str:
        """MD5 hash of normalised (lowercase, collapsed whitespace) text."""
        normalised = " ".join(text.lower().split())
        return hashlib.md5(normalised.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def get_synthesis_stats(self, state: AgentState) -> Dict[str, Any]:
        """Return synthesis metadata from state (for UI / debugging)."""
        return state.metadata.get("synthesis", {})
