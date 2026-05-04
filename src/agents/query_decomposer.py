"""
Query Decomposer Agent - Tactical Level 2

New flow (v2):
  1. Ask LLM: does this query need decomposition? (yes/no)
  2. If yes  → decompose with _decompose_multihop() prompt (unchanged)
  3. If no   → sub_queries = [original_query]

The agent no longer depends on state.strategy set by the Planner.
It runs *before* the Planner in the workflow so that the Planner can
assign per-sub-query retrieval strategies.
"""

import json
import re
from typing import List

from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Strategy
from src.config import get_settings
from src.utils.llm_content import message_content_to_text


class QueryDecomposer(BaseAgent):


    def __init__(self, llm: ChatOpenAI = None):
        super().__init__(name="query_decomposer", version="2.0.0")
        if llm is not None:
            self.llm = llm
        else:
            settings = get_settings()
            self.llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> AgentState:
        """
        Determine whether to decompose the query and produce sub_queries.

        Does NOT read state.strategy — makes its own decision via LLM.
        Sets state.strategy as a side-effect for backward compatibility.
        """
        query = state.query
        self.log(f"Evaluating query for decomposition: {query[:80]}", level="info")

        # ── Step 1: LLM decides whether decomposition is needed ────────
        need_decompose, decision_reason = self._should_decompose_with_llm(query)
        self.log(
            f"Decomposition decision: {'YES' if need_decompose else 'NO'} "
            f"— {decision_reason}",
            level="info",
        )

        # ── Step 2: Decompose or pass through ──────────────────────────
        if need_decompose:
            self.log("Decomposing query into sub-queries…", level="info")
            sub_queries = self._decompose_multihop(query)
            # Safety: if decomposition somehow returned only the original
            # query, treat it as no decomposition
            if len(sub_queries) == 1 and sub_queries[0].strip() == query.strip():
                need_decompose = False
                strategy = Strategy.SIMPLE
            else:
                strategy = Strategy.DECOMPOSE
        else:
            self.log("No decomposition needed — using original query", level="info")
            sub_queries = [query]
            strategy = Strategy.SIMPLE

        # ── Step 3: Write outputs ──────────────────────────────────────
        state.sub_queries = sub_queries
        # Set strategy for backward-compatibility (Planner may override later)
        state.strategy = strategy

        state.metadata["decomposition"] = {
            "original_query":  query,
            "need_decompose":  need_decompose,
            "decision_reason": decision_reason,
            "strategy":        strategy.value if hasattr(strategy, "value") else str(strategy),
            "sub_query_count": len(sub_queries),
            "sub_queries":     sub_queries,
        }

        self.log(
            f"Decomposer result: {len(sub_queries)} sub-queries "
            f"(strategy={strategy})",
            level="info",
        )
        return state

    # ------------------------------------------------------------------
    # LLM-based decomposition decision
    # ------------------------------------------------------------------

    def _should_decompose_with_llm(self, query: str) -> tuple[bool, str]:
        """
        Ask the LLM whether the query should be decomposed.

        Returns
        -------
        (need_decompose: bool, reason: str)
        """
        prompt = f"""你是一个 RAG 系统的查询分析模块。
判断以下用户问题是否需要拆解成多个子问题来分别检索,才能完整回答。

用户问题: "{query}"

判断标准：
- 需要拆解(true)：问题涉及多个独立方面、需要跨领域比较、需要链式推理、
  包含"以及/同时/和...的区别/优缺点/关系"等复合结构
- 不需要拆解(false)：单一概念定义、单一事实查询、直接命名解释、
  "什么是X"/"为什么叫X"/"X是什么"等直接问题

只输出 JSON,不要任何解释或代码块标记:
{{"need_decompose": true或false, "reason": "一句话说明原因"}}"""

        try:
            response = self.llm.invoke(prompt)
            raw = message_content_to_text(response).strip()
            self.log(f"Decompose-decision LLM response: {raw[:200]}", level="debug")

            # Strip markdown fences
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON in response: {raw[:100]}")

            data = json.loads(match.group())
            need = bool(data.get("need_decompose", False))
            reason = str(data.get("reason", ""))
            return need, reason

        except Exception as exc:
            self.log(
                f"LLM decompose-decision failed ({exc}), "
                "falling back to rule-based check",
                level="warning",
            )
            # Rule-based fallback
            need = not self._is_direct_definition_query(query)
            return need, "fallback-rules"

    # ------------------------------------------------------------------
    # Decomposition (prompt unchanged from v1)
    # ------------------------------------------------------------------

    def _decompose_multihop(self, query: str) -> List[str]:
        """
        Break a complex query into 2-4 focused sub-queries for retrieval.

        Sub-queries should stay natural enough for entity extraction. Keyword
        fragments can work for vector/BM25 retrieval but often break NER-backed
        graph retrieval.
        """
        prompt = f"""You are a query decomposition module for a retrieval system.
Your goal is to break a complex question into a minimal set of high-quality sub-queries for retrieval.

Original Query: {query}

Strict requirements:
1. Output ONLY 2-4 sub-queries. Never exceed 4.
2. Each sub-query must be a complete, natural-language question, not a keyword fragment.
3. Each sub-query should target a single aspect (no multi-aspect mixing).
4. Avoid semantic overlap between sub-queries.
5. Do NOT include summarization or conclusion-type queries.
6. Preserve the original core entity names, acronyms, product names, method names, and paper-specific terms in every sub-query where they are relevant.
7. Keep sub-queries concise, but do not remove grammar words if doing so makes the query unnatural.
8. Ensure sub-queries are diverse enough to cover different perspectives of the original question.

Bad examples:
- "AGCD two phases names"
- "first phase mechanism in AGCD"
- "second phase mechanism in AGCD"

Good examples:
- "What are the two phases of AGCD called?"
- "What happens during the first phase of AGCD?"
- "What happens during the second phase of AGCD?"

Optimization priorities:
- Minimize total query count
- Maximize information coverage
- Improve retrieval precision across vector, keyword, and graph retrieval
- Keep each query natural enough for named-entity recognition

Output format — return ONLY a JSON array, no explanation:
[
  "sub query 1",
  "sub query 2"
]"""

        try:
            response = self.llm.invoke(prompt)
            sub_queries = self._parse_sub_queries(message_content_to_text(response))
            self.log(
                f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}",
                level="info",
            )
            return sub_queries
        except Exception as exc:
            self.log(f"Decomposition failed: {exc}", level="error")
            return [query]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_sub_queries(self, text: str) -> List[str]:
        
        # ── 1. Try JSON array ──────────────────────────────────────────
        json_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if json_match:
            try:
                candidates = json.loads(json_match.group())
                result = [str(q).strip() for q in candidates if str(q).strip()]
                if result:
                    return result[:4]
            except (json.JSONDecodeError, TypeError):
                pass

        # ── 2. Numbered list fallback ──────────────────────────────────
        sub_queries = []
        for line in text.strip().split('\n'):
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line)
            if match:
                sub_queries.append(match.group(1).strip())
        if sub_queries:
            return sub_queries[:4]

        # ── 3. Last resort: non-empty lines ───────────────────────────
        return [line.strip() for line in text.strip().split('\n') if line.strip()][:4]

    def _is_direct_definition_query(self, query: str) -> bool:
        """
        Rule-based fallback: return True for simple definition/naming
        questions that should NOT be decomposed.
        """
        q = " ".join(query.lower().split())
        simple_patterns = [
            r"^what is\b",
            r"^what are\b",
            r"^what does .+ mean\??$",
            r"^why (is|are|was|were) .+ called .+\??$",
            r"^why (is|are|was|were) .+ known as .+\??$",
            r"^define\b",
            r"^explain what\b",
        ]
        complex_markers = [
            " compare ", " contrast ", " difference between ",
            " relationship between ", " how does ", " impact ",
            " depend ", " cause ", " versus ", " vs ",
        ]
        padded = f" {q} "
        return (
            any(re.search(pattern, q) for pattern in simple_patterns)
            and not any(marker in padded for marker in complex_markers)
        )
