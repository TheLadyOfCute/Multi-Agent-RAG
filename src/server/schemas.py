"""Chat request schemas."""

from __future__ import annotations

from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

class RagasRequest(BaseModel):
    test_file: str = "data/test_questions.json"
    reuse_rag_outputs: bool = False
