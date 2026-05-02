from typing import List
from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import ValidationError
from src.config import get_settings
from src.utils.llm_content import message_content_to_text


class ValidatorAgent(BaseAgent):
    
    def __init__(
        self,
        llm: ChatOpenAI,
        threshold: float = None,
        max_retries: int = None
    ):
        super().__init__(name="validator", version="1.0.0")
        
        self.llm = llm
        
        # Load settings from config
        settings = get_settings()
        self.threshold = (
            threshold 
            if threshold is not None 
            else settings.validator_threshold
        )
        self.max_retries = (
            max_retries 
            if max_retries is not None 
            else settings.validator_max_retries
        )
        self.log(
            f"Initialized with threshold={self.threshold}, "
            f"max_retries={self.max_retries}",
            level="debug"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        
        try:
            query = state.query
            chunks = state.chunks
            current_round = state.retrieval_round
            
            self.log(
                f"Validating {len(chunks)} chunks (round {current_round})",
                level="info"
            )
            
            # Step 1: Calculate sufficiency score
            score = self._calculate_sufficiency(query, chunks, state.total_docs)
            state.validation_score = score
            
            self.log(f"Sufficiency score: {score:.3f}", level="info")
            
            # Step 2: Make decision
            decision = self._make_decision(score, current_round)
            state.validation_status = decision
            
            self.log(
                f"Decision: {decision} (score: {score:.3f}, "
                f"round: {current_round}/{self.max_retries})",
                level="info"
            )
            
            # Step 3: Add metadata
            state.metadata["validator"] = {
                "score": score,
                "decision": decision,
                "retrieval_round": current_round,
                "threshold": self.threshold,
                "max_retries": self.max_retries
            }
            
            return state
            
        except Exception as e:
            self.log(f"Validation failed: {str(e)}", level="error")
            raise ValidationError(
                message=f"Chunk validation failed: {str(e)}",
                validation_type="sufficiency",
                details={
                    "query": state.query,
                    "chunk_count": len(state.chunks)
                }
            ) from e
    
    def _calculate_sufficiency(self, query: str, chunks: List[Chunk], total_docs: int = 0) -> float:

        if not chunks:
            self.log("No chunks to validate", level="warning")
            return 0.0

        try:
            # Factor 1: Relevance (50%)
            relevance_score = self._check_relevance(query, chunks)

            # Factor 2: Coverage (30%)
            coverage_score = self._check_coverage(query, chunks, total_docs)
            
            # Factor 3: Confidence (20%)
            confidence_score = self._check_confidence(chunks)
            
            self.log(
                f"Validation factors: relevance={relevance_score:.2f}, "
                f"coverage={coverage_score:.2f}, confidence={confidence_score:.2f}",
                level="debug"
            )
            
            # Weighted combination
            final_score = (
                relevance_score * 0.5 +
                coverage_score * 0.3 +
                confidence_score * 0.2
            )
            
            return max(0.0, min(final_score, 1.0))
            
        except Exception as e:
            self.log(f"Error calculating sufficiency: {str(e)}", level="warning")
            # Return conservative score on error
            return 0.5
    
    def _check_relevance(self, query: str, chunks: List[Chunk]) -> float:

        # Prepare context
        context = "\n\n".join([
            f"Chunk {i+1} (score: {chunk.score:.2f}):\n"
            f"{chunk.text}"
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""You are evaluating whether the retrieved chunks are relevant to the user's query.

Query:
"{query}"

Retrieved Chunks:
{context}

Evaluate relevance based on:
1. Whether the chunks address the main intent of the query.
2. Whether they contain information that would help answer the query.
3. Whether they cover the key entities, constraints, time period, or conditions in the query.
4. Whether the chunks are specific enough, not merely topically related.

Scoring guide:
- 0.0-0.2: Completely irrelevant. No meaningful connection to the query.
- 0.2-0.4: Weakly relevant. Shares topic or keywords, but does not help answer the query.
- 0.4-0.6: Partially relevant. Some useful information, but misses important parts of the query.
- 0.6-0.8: Relevant. Provides useful information that addresses most of the query.
- 0.8-1.0: Highly relevant. Directly answers the query or contains the key evidence needed to answer it.

Important:
- Do not reward keyword overlap alone.
- Penalize chunks that discuss the same topic but answer a different question.
- If the chunks are insufficient to answer the query, do not score above 0.7.
- If the chunks directly answer the query, score at least 0.8.

Respond with ONLY one number between 0.0 and 1.0."""

        try:
            response = self.llm.invoke(prompt)
            score_text = message_content_to_text(response).strip()
            
            # Extract number
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(score, 1.0))
            else:
                self.log(
                    f"Could not parse relevance score: {score_text}, using fallback",
                    level="warning"
                )
                return self._fallback_relevance_score(query, chunks)
                
        except Exception as e:
            self.log(
                f"LLM relevance check failed: {str(e)}, using fallback",
                level="warning"
            )
            return self._fallback_relevance_score(query, chunks)
    
    def _fallback_relevance_score(self, query: str, chunks: List[Chunk]) -> float:
        """
        Fallback relevance scoring without LLM.
        
        Uses average chunk scores as proxy for relevance.
        
        Args:
            query: User query string
            chunks: List of retrieved chunks
        
        Returns:
            Fallback relevance score
        """
        if not chunks:
            return 0.0
        
        # Use average of existing chunk scores
        scores = [c.score for c in chunks if c.score is not None]
        
        if scores:
            return sum(scores) / len(scores)
        else:
            # Default to moderate if no scores
            return 0.5
    
    def _check_coverage(self, query: str, chunks: List[Chunk], total_docs: int = 0) -> float:
        """
        评估检索到的文档块对查询的覆盖程度。

        综合考虑两个维度：
        1. 数量覆盖 — 文档块数量是否足够回答查询中的各个子问题/方面
        2. 来源多样性 — 文档块是否来自多个不同的文档，避免信息来源单一

        Args:
            query: 用户的查询文本
            chunks: 检索到的文档块列表
            total_docs: 知识库中的总文档数，用于相对多样性评估

        Returns:
            覆盖度评分，范围 [0.0, 1.0]，越高表示覆盖越充分
        """
        if not chunks:
            return 0.0

        # 将查询转为小写，便于后续关键词匹配
        query_lower = query.lower()

        # 统计查询中的问号数量（每个问号通常对应一个子问题）
        question_marks = query.count("?")
        # 统计 "and"/"or" 连接词数量（每个连接词暗示查询包含多个方面）
        and_or_count = query_lower.count(" and ") + query_lower.count(" or ")

        # 根据问号和连接词估算查询包含的方面数，至少为 1
        num_aspects = max(1, question_marks + and_or_count)

        # 当前检索到的文档块总数
        chunk_count = len(chunks)

        # 启发式规则：每个查询方面至少需要 2 个文档块来充分回答
        ideal_chunks = num_aspects * 2
        # 计算数量覆盖比，上限为 1.0（超过理想数量不再加分）
        coverage_ratio = min(chunk_count / ideal_chunks, 1.0)

        # 计算来源多样性：统计检索命中了多少个不同文档
        unique_docs = len(set(c.doc_id for c in chunks if c.doc_id))
        # 多样性基准取实际文档数和查询方面数的较小值，至少为 1
        # 有 chunks 必然有文档，total_docs >= 1
        diversity_baseline = max(1, min(total_docs, num_aspects, 3))
        # 无文档信息时给 0.5 的基础分
        diversity_score = min(unique_docs / diversity_baseline, 1.0) if unique_docs > 0 else 0.5

        # 加权合成最终覆盖度评分：数量覆盖占 70%，来源多样性占 30%
        coverage_score = (coverage_ratio * 0.7) + (diversity_score * 0.3)

        return coverage_score
    
    def _check_confidence(self, chunks: List[Chunk]) -> float:
        """
        Check confidence in chunk quality.
        
        Based on:
        - Average chunk scores (higher = better)
        - Score consistency (less variance = better)
        - Minimum score threshold
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Confidence score (0.0-1.0)
        """
        if not chunks:
            return 0.0
        
        #c.score在reranker阶段已经被归一化到0-1范围内
        scores = [c.score for c in chunks if c.score is not None]
        
        if not scores:
            return 0.5  # Moderate confidence if no scores
        
        # Average score
        avg_score = sum(scores) / len(scores)
        
        # Minimum score (worst chunk)
        min_score = min(scores)
        
        # Score variance (consistency)
        if len(scores) > 1:
            #方差越小，说明分数越一致，置信度越高
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency = max(0.0, 1.0 - variance)
        else:
            consistency = 1.0
        
        # Weighted combination
        confidence = (
            avg_score * 0.5 +
            min_score * 0.3 +
            consistency * 0.2
        )
        
        return confidence
    
    def _make_decision(self, score: float, current_round: int) -> str:
        """
        Make validation decision based on score and retry count.
        
        Decision Logic:
        1. If score >= threshold → PROCEED
        2. If score < threshold AND retries available → RETRIEVE_MORE
        3. If max retries reached → PROCEED (force)
        
        Args:
            score: Sufficiency score
            current_round: Current retrieval round (0-indexed)
        
        Returns:
            "PROCEED" or "RETRIEVE_MORE"
        
        Example:
            >>> decision = validator._make_decision(0.85, 0)
            >>> print(decision)  # "PROCEED"
            
            >>> decision = validator._make_decision(0.45, 0)
            >>> print(decision)  # "RETRIEVE_MORE"
            
            >>> decision = validator._make_decision(0.45, 2)  # max retries
            >>> print(decision)  # "PROCEED"
        """
        # Check if score meets threshold
        if score >= self.threshold:
            return "PROCEED"
        
        # Check if we can retry
        if current_round < self.max_retries:
            self.log(
                f"Score {score:.3f} below threshold {self.threshold}, "
                f"triggering retry (round {current_round + 1}/{self.max_retries})",
                level="warning"
            )
            return "RETRIEVE_MORE"
        
        # Max retries reached, force proceed
        self.log(
            f"Max retries reached ({self.max_retries}), "
            f"proceeding with score {score:.3f}",
            level="warning"
        )
        return "PROCEED"
    
