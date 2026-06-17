"""
Critic Agent - Tactical Level 2 Agent.

Reviews generated answers for quality and provides improvement feedback.
Triggers regeneration if quality is below threshold.
"""

import json
from typing import List, Dict, Any, Optional
from enum import Enum

from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState
from src.config import get_settings
from src.utils.exceptions import AgenticRAGException
from src.utils.llm_content import message_content_to_text


class CriticDecision(Enum):
    """Critic's decision on answer quality."""
    APPROVED = "approved"
    REGENERATE = "regenerate"
    INSUFFICIENT_INFO = "insufficient_info"


class CriticError(AgenticRAGException):
    """Error during answer critique."""
    pass


class CriticAgent(BaseAgent):

    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        quality_threshold: float = 0.7,
        max_iterations: int = 3
    ):
        
        super().__init__(name="critic", version="1.0.0")
        
        settings = get_settings()
        
        # Initialize LLM
        if llm is None:
            self.llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0.0,  # Deterministic for consistency
                max_tokens=2000,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
                extra_body={"enable_thinking": False},
            )
        else:
            self.llm = llm
        
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        
        self.log(
            f"Initialized with threshold={quality_threshold:.2f}, "
            f"max_iterations={max_iterations}",
            level="info"
        )
    
    def execute(self, state: AgentState) -> AgentState:

        try:
            query = state.query
            answer = state.answer
            chunks = state.chunks
            
            if not answer:
                self.log("No answer to critique", level="warning")
                state.critic_decision = CriticDecision.INSUFFICIENT_INFO
                state.critic_score = 0.0
                return state
            
            self.log(
                f"Critiquing answer for: {query[:50]}...",
                level="info"
            )
            
            # Perform critique
            critique_result = self._critique_answer(query, answer, chunks)
            
            # Update state
            state.critic_score = critique_result['overall_score']
            state.critic_feedback = critique_result['feedback']
            state.critic_scores = critique_result['scores']
            
            # Make decision
            decision = self._make_decision(critique_result['overall_score'])
            state.critic_decision = decision
            
            # Add metadata
            state.metadata["critic"] = {
                "overall_score": critique_result['overall_score'],
                "decision": decision.value,
                "scores": critique_result['scores'],
                "iteration": state.metadata.get("regeneration_count", 0)
            }
            
            self.log(
                f"Critique complete: score={critique_result['overall_score']:.3f}, "
                f"decision={decision.value}",
                level="info"
            )
            
            return state
            
        except Exception as e:
            self.log(f"Critique failed: {str(e)}", level="error")
            raise CriticError(
                message=f"Failed to critique answer: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _critique_answer(
        self,
        query: str,
        answer: str,
        chunks: List
    ) -> Dict[str, Any]:
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):  # Use top 5
            context_parts.append(f"[{i}] {chunk.text}")
        
        context = "\n".join(context_parts)
        
        # Create critique prompt
        prompt = f"""You are a quality reviewer for AI-generated answers. Your job is to critique an answer based on provided context.

User Query: {query}

Available Context:
{context}

Generated Answer:
{answer}

Evaluate the answer on these criteria (score each 0.0-1.0):

1. ACCURACY: Does the answer provide correct information based on the context?
2. COMPLETENESS: Does it fully address all parts of the query?
3. CITATIONS: Are citations properly used and relevant?
4. CLARITY: Is the answer well-structured and easy to understand?
5. RELEVANCE: Does it directly answer the question asked?

Respond with a raw JSON object (not markdown wrapped) strictly using this schema:
{{
  "scores": {{
    "accuracy": [float 0.0-1.0],
    "completeness": [float 0.0-1.0],
    "citations": [float 0.0-1.0],
    "clarity": [float 0.0-1.0],
    "relevance": [float 0.0-1.0]
  }},
  "feedback": "Specific suggestions for improvement, or 'No improvements needed' if excellent",
  "recommendation": "APPROVED" or "REGENERATE"
}}"""
        
        # Get critique from LLM
        try:
            response = self.llm.invoke(prompt)
            critique_text = message_content_to_text(response)
            
            # Parse response
            parsed = self._parse_critique(critique_text)
            
            return parsed
            
        except Exception as e:
            raise CriticError(
                message=f"LLM critique failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def _parse_critique(self, critique_text: str) -> Dict[str, Any]:
        """
        Parse LLM critique response (expected JSON).

        Args:
            critique_text: Raw LLM response

        Returns:
            Parsed scores, feedback, and recommendation
        """
        import re

        # Strip markdown code fences if LLM wrapped them anyway
        cleaned = re.sub(r"```(?:json)?\s*", "", critique_text).strip()
        cleaned = cleaned.rstrip("`").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            raise CriticError(
                message="LLM returned invalid JSON for critique",
                details={"raw_response": critique_text[:500]},
            )

        # Validate and clamp scores
        required_keys = {"accuracy", "completeness", "citations", "clarity", "relevance"}
        raw_scores = data.get("scores", {})
        scores = {}
        for key in required_keys:
            val = raw_scores.get(key, 0.5)
            scores[key] = max(0.0, min(1.0, float(val)))

        feedback = str(data.get("feedback", ""))
        recommendation = str(data.get("recommendation", "REGENERATE")).upper()

        # Calculate overall score (weighted average)
        weights = {
            'accuracy': 0.3,
            'completeness': 0.25,
            'citations': 0.15,
            'clarity': 0.15,
            'relevance': 0.15
        }

        overall_score = sum(
            scores[criterion] * weight
            for criterion, weight in weights.items()
        )

        return {
            'scores': scores,
            'overall_score': overall_score,
            'feedback': feedback,
            'recommendation': recommendation,
        }
    
    def _make_decision(
        self,
        overall_score: float,
    ) -> CriticDecision:
        # Check if score meets threshold
        if overall_score >= self.quality_threshold:
            return CriticDecision.APPROVED
        else:
            return CriticDecision.REGENERATE
    
