"""RAGAS evaluator compatibility layer.

The project currently pins ``ragas==0.1.16``. This module keeps the public
evaluation surface small and normalizes metric names so the rest of the app
does not depend on a single RAGAS release's naming details.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from langchain_core._api import LangChainDeprecationWarning
except Exception:  # pragma: no cover - fallback for package layout changes
    LangChainDeprecationWarning = Warning

warnings.filterwarnings(
    "ignore",
    category=LangChainDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*LangChain uses pydantic v2 internally.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*pydantic_v1 module was a compatibility shim.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)

# ragas, datasets, and related heavy deps are imported lazily (inside methods)
# to avoid asyncio event-loop conflicts on Windows.
# Importing ragas at module level can initialize event-loop-heavy dependencies.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import get_settings

# Kept as a patch target for legacy tests that mocked the old Anthropic path.
ChatAnthropic = ChatOpenAI
evaluate = None


def _import_ragas_metrics():
    """Lazily import ragas metrics to defer asyncio initialization."""
    from ragas.metrics import context_precision, context_recall, faithfulness
    try:
        from ragas.metrics import answer_relevancy as relevancy_metric
    except ImportError:
        from ragas.metrics import response_relevancy as relevancy_metric
    return faithfulness, context_precision, context_recall, relevancy_metric

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

class RAGASEvaluator:
    """Evaluate RAG responses with LLM-based RAGAS metrics."""

    def __init__(
        self,
        llm: Any = None,
        embeddings: Any = None,
        model: Optional[str] = None,
        run_config: Optional[Any] = None,
    ):
        self.llm = llm if llm is not None else self._create_default_llm(model)
        self.embeddings = (
            embeddings if embeddings is not None else self._create_default_embeddings()
        )
        # Lazy import: defer ragas asyncio initialization until first use.
        from ragas.run_config import RunConfig
        faithfulness, context_precision, context_recall, relevancy_metric = (
            _import_ragas_metrics()
        )
        # max_wait 设为 300s，避免慢速 LLM 调用被误判为超时
        self.run_config = run_config or RunConfig(
            max_workers=1, max_retries=3, max_wait=300, timeout=300
        )
        self.metrics = [
            faithfulness,
            relevancy_metric,
            context_precision,
            context_recall,
        ]

    def _create_default_llm(self, model: Optional[str] = None) -> ChatOpenAI:
        settings = get_settings()
        return ChatOpenAI(
            model=model or settings.llm_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0.0,
            # Qwen3 系列模型默认开启 thinking，但 RAGAS 会用 n>1 生成多个问题变体，
            # 与 enable_thinking=true 冲突（API 要求 n=1）。
            # 通过 model_kwargs 强制关闭 thinking，同时锁定 n=1。
            model_kwargs={
                "extra_body": {"enable_thinking": False},
            },
        )

    def _create_default_embeddings(self) -> OpenAIEmbeddings:
        settings = get_settings()
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            dimensions=settings.embedding_dimension,
            check_embedding_ctx_length=False,
        )

    def evaluate_single_case(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, Optional[float]]:
        """Evaluate one RAG output and return normalized metric names."""
        from datasets import Dataset
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer or ""],
                "contexts": [contexts or []],
                "ground_truth": [ground_truth or ""],
            }
        )
        scores: Dict[str, Optional[float]] = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }
        metric_errors: Dict[str, str] = {}

        try:
            official_scores = self._evaluate_with_ragas(dataset, self.metrics)
            scores.update(self._normalize_scores(official_scores))
        except Exception as exc:  # noqa: BLE001 - best-effort metrics are required
            metric_errors["official_metrics"] = str(exc)

        scores["overall"] = _mean(scores.get(name) for name in METRIC_NAMES)
        scores["metric_errors"] = metric_errors
        return scores

    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dict[str, Optional[float]]:
        """Evaluate a batch and return mean scores by metric."""
        per_case = [
            self.evaluate_single_case(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=g,
            )
            for q, a, c, g in zip(questions, answers, contexts, ground_truths)
        ]
        scores = {
            name: _mean(row.get(name) for row in per_case)
            for name in METRIC_NAMES
        }
        scores["overall"] = _mean(scores.values())
        return scores

    def _evaluate_with_ragas(self, dataset: Any, metrics: List[Any]) -> Any:
        global evaluate
        if evaluate is None:
            from ragas import evaluate as ragas_evaluate

            evaluate = ragas_evaluate
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pydantic_v1.*")
            return evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=self.run_config,
                raise_exceptions=False,
            )

    def _normalize_scores(self, result: Any) -> Dict[str, Optional[float]]:
        raw = self._result_to_dict(result)
        return {
            "faithfulness": _as_float(raw.get("faithfulness")),
            "answer_relevancy": _as_float(self._extract_relevancy_score(raw)),
            "context_precision": _as_float(raw.get("context_precision")),
            "context_recall": _as_float(raw.get("context_recall")),
        }

    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            return df.iloc[0].to_dict() if len(df) else {}
        return {}

    def _extract_relevancy_score(self, result: Any) -> Optional[float]:
        raw = self._result_to_dict(result) if not isinstance(result, dict) else result
        answer_relevancy = raw.get("answer_relevancy")
        if answer_relevancy is None:
            answer_relevancy = raw.get("response_relevancy")
        return _as_float(answer_relevancy)



def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)
