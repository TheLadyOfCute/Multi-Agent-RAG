"""
Agent State Model - Shared state between all agents.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum
from src.models.chunk import Chunk 

class Strategy(str, Enum):
    """Query execution strategies"""
    SIMPLE = "simple"
    DECOMPOSE = "decompose"

class AgentState(BaseModel):
    """Shared state passed between agents"""
    
    # Input
    query: str
    
    # Planner outputs
    complexity: Optional[float] = Field(None, ge=0.0, le=1.0)
    strategy: Optional[Strategy] = None

    # Planner: Query Router outputs
    # ["vector", "keyword", "graph"]
    selected_retrievers: List[str] = Field(default_factory=list)
    # 每个检索器分配的 top-k 配额，e.g. {"vector": 10, "keyword": 10}
    retriever_quotas: Dict[str, int] = Field(default_factory=dict)

    # Query Decomposer field
    sub_queries: Optional[List[str]] = None

    # Per-sub-query retrieval plans (set by Planner after decomposition).
    # Each entry: {"query": str, "retrievers": List[str], "quotas": Dict[str, int]}
    sub_query_plans: Optional[List[Dict[str, Any]]] = None
    
    # Retrieval outputs
    chunks: List[Chunk] = Field(default_factory=list)
    retrieval_round: int = Field(default=0, ge=0)
    
    # Validator outputs
    validation_status: Optional[str] = None
    validation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Generator outputs
    answer: Optional[str] = None
    
    # Critic outputs
    critic_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    critic_feedback: Optional[str] = None
    critic_scores: Optional[Dict[str, float]] = None
    critic_decision: Optional[Any] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("strategy", mode="before")
    def normalize_strategy(cls, value):
        """Map legacy strategy values onto the new two-state model."""
        if value in (None, ""):
            return None
        if isinstance(value, Strategy):
            return value
        normalized = str(value).strip().lower()
        if normalized == "simple":
            return Strategy.SIMPLE
        if normalized in {"decompose", "multihop", "graph"}:
            return Strategy.DECOMPOSE
        return value
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
    
