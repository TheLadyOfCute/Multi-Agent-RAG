"""
Planner Agent - Query Router / Strategy Selector (v3).

New flow (v3)
-------------
The Planner now runs *after* the QueryDecomposer. It receives
``state.sub_queries`` (already produced) and assigns a retrieval
strategy to each sub-query in a single LLM call.

Outputs
-------
state.sub_query_plans  : List[Dict] — per-sub-query retriever config
    [
      {"query": "...", "retrievers": ["vector"], "quotas": {"vector": 10}},
      {"query": "...", "retrievers": ["keyword", "graph"], "quotas": {...}},
    ]
state.selected_retrievers : union of all retrievers (backward-compat)
state.retriever_quotas    : max quota per retriever  (backward-compat)
state.strategy            : SIMPLE / DECOMPOSE       (backward-compat)
state.complexity          : float                    (backward-compat)
"""

import re
import json
from typing import Dict, List, Any

from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Strategy
from src.utils.exceptions import AgentExecutionError
from src.config import get_settings
from src.utils.llm_content import message_content_to_text


# Default / max quota assigned to each selected retriever
_DEFAULT_QUOTA = 10
_MAX_QUOTA     = 15
_VALID_RETRIEVERS = {"vector", "keyword", "graph"}


class PlannerAgent(BaseAgent):
    """
    Planner Agent — per-sub-query retrieval strategy selector.

    Runs after QueryDecomposer. Reads ``state.sub_queries`` and
    assigns a retrieval plan to each sub-query in one LLM call.

    Example
    -------
    >>> state.sub_queries = ["RAG retrieval mechanism", "RAG accuracy improvement"]
    >>> result = planner.run(state)
    >>> print(result.sub_query_plans)
    [
      {"query": "RAG retrieval mechanism",
       "retrievers": ["vector", "keyword"],
       "quotas": {"vector": 10, "keyword": 10}},
      {"query": "RAG accuracy improvement",
       "retrievers": ["vector"],
       "quotas": {"vector": 10}},
    ]
    """

    def __init__(self, llm: ChatOpenAI, **kwargs):
        super().__init__(name="planner", version="3.0.0")
        self.llm      = llm
        self.settings = get_settings()
        self.log("Initialized as per-sub-query retrieval strategy selector", level="debug")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> AgentState:
        """
        Assign a retrieval plan to each sub-query.

        Reads  : state.sub_queries  (set by QueryDecomposer)
        Writes : state.sub_query_plans
                 state.selected_retrievers  (union, backward-compat)
                 state.retriever_quotas     (max per name, backward-compat)
                 state.strategy             (backward-compat)
                 state.complexity           (backward-compat)
        """
        try:
            sub_queries = state.sub_queries or [state.query]
            self.log(
                f"Planning retrieval for {len(sub_queries)} sub-queries",
                level="info",
            )

            # ── Step 1: LLM assigns retrievers to every sub-query ──────
            plans = self._plan_for_sub_queries(sub_queries)

            # ── Step 2: Write primary output ───────────────────────────
            state.sub_query_plans = plans

            # ── Step 3: Backward-compat aggregated fields ──────────────
            all_retrievers: List[str] = []
            union_quotas:   Dict[str, int] = {}
            for p in plans:
                for name in p["retrievers"]:
                    if name not in all_retrievers:
                        all_retrievers.append(name)
                    q = p["quotas"].get(name, _DEFAULT_QUOTA)
                    union_quotas[name] = max(union_quotas.get(name, 0), q)

            state.selected_retrievers = all_retrievers
            state.retriever_quotas    = union_quotas

            # Infer strategy from whether decomposition produced >1 query
            state.strategy = (
                Strategy.DECOMPOSE if len(sub_queries) > 1 else Strategy.SIMPLE
            )
            state.complexity = self._infer_complexity(all_retrievers)

            self.log(
                f"Planner v3: {len(plans)} plans, "
                f"union_retrievers={all_retrievers}, strategy={state.strategy}",
                level="info",
            )

            # ── Step 4: Metadata ───────────────────────────────────────
            state.metadata["planner"] = {
                "version":             "3.0.0 (per-sub-query-planner)",
                "sub_query_count":     len(sub_queries),
                "sub_query_plans":     plans,
                "selected_retrievers": all_retrievers,
                "retriever_quotas":    union_quotas,
                "strategy":            state.strategy,
                "complexity":          state.complexity,
            }

            return state

        except Exception as exc:
            self.log(f"Planning failed: {exc}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Failed to plan retrieval: {exc}",
                details={"query": state.query},
            ) from exc

    # ------------------------------------------------------------------
    # LLM-driven per-sub-query planning
    # ------------------------------------------------------------------

    def _plan_for_sub_queries(
        self, sub_queries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        One LLM call to assign retrievers + quotas to every sub-query.

        Returns
        -------
        List of plan dicts:
            {"query": str, "retrievers": List[str], "quotas": Dict[str,int]}
        """
        prompt = self._build_multi_query_prompt(sub_queries)

        try:
            response = self.llm.invoke(prompt)
            raw      = message_content_to_text(response).strip()
            self.log(f"LLM plan response: {raw[:300]}", level="debug")
            plans = self._parse_plans(raw, sub_queries)
            return plans

        except Exception as exc:
            self.log(
                f"LLM planning failed ({exc}), using rule-based fallback",
                level="warning",
            )
            return self._fallback_plans(sub_queries)

    def _build_multi_query_prompt(self, sub_queries: List[str]) -> str:
        """Build the prompt that requests a retrieval plan for every sub-query."""
        numbered = "\n".join(
            f'{i + 1}. "{q}"' for i, q in enumerate(sub_queries)
        )
        example_out = json.dumps(
            [
                {
                    "query":     sub_queries[0],
                    "retrievers": ["vector"],
                    "quotas":    {"vector": 10},
                }
            ],
            ensure_ascii=False,
            indent=2,
        )

        return f"""你是一个 RAG 检索策略规划器。
根据下方问题列表，为每个问题选择最合适的检索器，并分配 top_k 检索数量。

问题列表:
{numbered}

可用检索器:
- vector  : 向量语义检索。擅长理解“相似性”、“为什么”、“是什么”、隐含意图和长段落概念。
- keyword : BM25 关键词检索。擅长精准匹配：产品型号(如 iPhone 15 Pro)、人名、特定编号、专有名词、报错代码。
- graph   : 知识图谱检索。擅长理清网状脉络：实体关系(A和B怎么认识的)、因果链条、上下级结构、多步推导。

对于每一个问题，请严格按照以下步骤进行思考并输出：
- 意图分析 (Analysis): 一句话分析该问题的核心诉求是什么？它最依赖哪种类型的信息？
- 检索器选择 (Selection): 结合分析，选择 1 到 3 个最合适的检索器。
- 额度分配 (Allocation): 设定总 top_k(10-15之间)，并根据重要性在选定的检索器中进行分配。

重要: 每个问题独立判断，同一组不同问题可以有不同的检索器组合。

只输出 JSON 数组，不要任何解释或代码块标记，格式如下:
{example_out}"""

    # 解析 LLM 返回的计划 JSON，并转换为校验后的计划列表
    def _parse_plans(
        self, raw: str, sub_queries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse the LLM JSON array response into validated plan dicts.

        Falls back to rule-based plans for individual entries that
        cannot be parsed.
        """

        # 去除 LLM 返回内容中可能包含的 markdown 代码块标记
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        # 从清理后的文本中提取 JSON 数组部分
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)

        # 如果没有找到 JSON 数组，则直接抛出异常
        if not match:
            raise ValueError(f"No JSON array in LLM response: {raw[:120]}")

        # 将匹配到的 JSON 字符串解析为 Python 对象
        raw_list = json.loads(match.group())

        # 校验解析结果必须是列表
        if not isinstance(raw_list, list):
            raise ValueError("LLM response is not a JSON array")

        # 存放最终生成的计划结果
        plans: List[Dict[str, Any]] = []

        # 遍历每个子查询，并为其生成对应的检索计划
        for i, sub_query in enumerate(sub_queries):
            # 默认当前计划项为空
            entry = None

            # 优先按位置从 LLM 返回列表中取对应计划项
            if i < len(raw_list):
                entry = raw_list[i]

            # 如果当前计划项不是字典，使用兜底规则生成计划
            if not isinstance(entry, dict):
                self.log(
                    f"Plan entry {i} is not a dict, using fallback",
                    level="warning",
                )
                plans.append(self._fallback_single_plan(sub_query))
                continue

            # 读取当前计划项中的 retrievers 字段
            raw_retrievers = entry.get("retrievers", [])

            # 如果 retrievers 是字符串，则转换成列表统一处理
            if isinstance(raw_retrievers, str):
                raw_retrievers = [raw_retrievers]

            # 清洗并校验 retriever 名称，只保留合法的检索器
            retrievers = [
                r.strip().lower()
                for r in raw_retrievers
                if r.strip().lower() in _VALID_RETRIEVERS
            ]

            # 如果没有合法检索器，则使用 vector 作为安全默认值
            if not retrievers:
                retrievers = ["vector"]

            # 读取当前计划项中的 quotas 字段
            raw_quotas = entry.get("quotas", {})

            # 存放每个检索器对应的配额
            quotas: Dict[str, int] = {}

            # 为每个检索器校验并生成检索配额
            for name in retrievers:
                # 获取当前检索器配额，不存在则使用默认配额
                q = raw_quotas.get(name, _DEFAULT_QUOTA)

                # 将配额转换为整数，转换失败则使用默认配额
                try:
                    q = int(q)
                except (TypeError, ValueError):
                    q = _DEFAULT_QUOTA

                # 将配额限制在允许范围内
                quotas[name] = max(1, min(q, _MAX_QUOTA))

            # 保存当前子查询对应的完整检索计划
            plans.append({
                "query":      sub_query,
                "retrievers": retrievers,
                "quotas":     quotas,
            })

        # 返回所有子查询对应的计划列表
        return plans

    # ------------------------------------------------------------------
    # Rule-based fallbacks
    # ------------------------------------------------------------------

    def _fallback_plans(self, sub_queries: List[str]) -> List[Dict[str, Any]]:
        """Rule-based fallback: one plan per sub-query."""
        return [self._fallback_single_plan(q) for q in sub_queries]

    def _fallback_single_plan(self, query: str) -> Dict[str, Any]:
        """Rule-based retriever selection for one query."""
        q = query.lower()
        relation_signals = [
            "关系", "影响", "依赖", "路径", "比较", "区别", "联系", "关联",
            "因果", "导致", "如何影响",
            "relation", "impact", "between", "compare", "difference",
            "how does", "effect of", "leads to", "related to",
        ]
        exact_signals = [
            '"', "'", "编号", "型号", "id", "版本", "#",
            "named", "called", "titled",
        ]
        has_relation = any(w in q for w in relation_signals)
        has_exact    = any(w in q for w in exact_signals)

        if has_relation:
            retrievers = ["graph", "vector"]
        elif has_exact:
            retrievers = ["keyword"]
        else:
            retrievers = ["vector"]

        return {
            "query":     query,
            "retrievers": retrievers,
            "quotas":    {r: _DEFAULT_QUOTA for r in retrievers},
        }

    # ------------------------------------------------------------------
    # Backward-compat helpers
    # ------------------------------------------------------------------

    def _infer_complexity(self, retrievers: List[str]) -> float:
        """Rough complexity proxy from number of distinct retrievers."""
        return {1: 0.2, 2: 0.5, 3: 0.8}.get(len(retrievers), 0.5)

