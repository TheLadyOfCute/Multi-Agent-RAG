"""Writer Agent - generate grounded answers with citations."""

from typing import Any, Dict, List, Optional
import re

from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.config import get_settings
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import AgenticRAGException
from src.utils.llm_content import message_content_to_text


class WriterError(AgenticRAGException):
    """Error during answer generation."""


class WriterAgent(BaseAgent):
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_tokens: int = None,
        temperature: float = None,
        include_sources: bool = True,
    ):
        super().__init__(name="writer", version="1.0.0")
        settings = get_settings()
        if llm is None:
            self.llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
            )
        else:
            self.llm = llm

        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature or settings.llm_temperature
        self.include_sources = include_sources

    def execute(self, state: AgentState) -> AgentState:
        try:
            query = state.query
            chunks = state.chunks
            if not chunks:
                state.answer = "I don't have enough information to answer this question."
                return state

            answer = self._generate_answer(query, chunks)
            formatted_answer = self._format_answer(answer, chunks)
            state.answer = formatted_answer
            state.metadata["writer"] = {
                "chunks_used": len(chunks),
                "answer_length": len(formatted_answer),
                "citations_count": self._count_citations(formatted_answer),
                "citation_ids": self._extract_citation_ids(formatted_answer),
                "citation_chunk_ids": self._extract_citation_chunk_ids(formatted_answer, chunks),
            }
            return state
        except Exception as e:
            raise WriterError(
                message=f"Failed to generate answer: {str(e)}",
                details={"query": state.query},
            ) from e

    def _generate_answer(self, query: str, chunks: List[Chunk]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):#1开始编号
            source = chunk.metadata.get("filename", "unknown")
            score = chunk.score if chunk.score else 0.0
            context_parts.append(f"[{i}] (Source: {source}, Relevance: {score:.2f})\n{chunk.text}\n")
        context = "\n".join(context_parts)

        prompt = f"""You are a precise and faithful assistant. Your ONLY job is to answer questions using the provided context. You must NEVER add information that is not explicitly stated in the context.

User Question: {query}

Context (with source references):
{context}

GROUNDING RULES (most important):
1. EVERY claim in your answer MUST come directly from the context above
2. Do NOT infer, assume, or add information beyond what the context states
3. If the context does not contain the answer, say:
   \"The provided documents do not contain information about [topic].\"
4. Do NOT use your general knowledge; ONLY the context matters

CITATION RULES:
1. Cite ONLY the specific chunk(s) that directly support EACH statement
2. Use inline citations: [1], [2], [3]
3. Each sentence should cite ONLY the chunks it actually uses

Answer (inline citations only, no Sources section):"""

        try:
            response = self.llm.invoke(prompt)
            return message_content_to_text(response)
        except Exception as e:
            raise WriterError(message=f"LLM generation failed: {str(e)}", details={"query": query}) from e

    def _format_answer(self, answer: str, chunks: List[Chunk]) -> str:
        """格式化答案，将引用标记转换为带来源信息的格式。"""
        # 如果未启用来源引用，直接返回原始答案
        if not self.include_sources:
            return answer

        # 从答案中提取所有引用标记 [1], [2] 等
        citations = re.findall(r"\[(\d+)\]", answer)
        unique_citations = sorted(set(int(c) for c in citations))
        # 没有引用标记则直接返回原始答案
        if not unique_citations:
            return answer

        # 按来源文件名对引用进行分组
        grouped_sources: Dict[str, List[int]] = {}
        for citation_num in unique_citations:
            # 确保引用编号在有效范围内
            if citation_num <= len(chunks):
                chunk = chunks[citation_num - 1]
                # 获取来源文件名，未知则标记为 "Unknown source"
                source = str(chunk.metadata.get("filename", "Unknown source"))
                grouped_sources.setdefault(source, []).append(citation_num)

        # 构建 Sources 部分，将同一来源的引用合并显示
        sources_section = "\n\n---\n\n**Sources:**\n"
        for source_key, citation_nums in grouped_sources.items():
            citation_label = ", ".join(str(num) for num in citation_nums)
            sources_section += f"\n[{citation_label}] {source_key}"
        return answer + sources_section

    def _count_citations(self, answer: str) -> int:
        citations = re.findall(r"\[(\d+)\]", answer)
        return len(set(citations))

    def _extract_citation_ids(self, answer: str) -> List[int]:
        citations = re.findall(r"\[(\d+)\]", answer)
        return sorted(set(int(citation) for citation in citations))

    def _extract_citation_chunk_ids(self, answer: str, chunks: List[Chunk]) -> List[str]:
        chunk_ids: List[str] = []
        for citation_num in self._extract_citation_ids(answer):
            if 1 <= citation_num <= len(chunks):
                chunk = chunks[citation_num - 1]
                chunk_ids.append(str(chunk.chunk_id))
        return chunk_ids

    def generate_with_feedback(self, query: str, chunks: List[Chunk], feedback: str) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("filename", "unknown")
            context_parts.append(f"[{i}] (Source: {source})\n{chunk.text}\n")
        context = "\n".join(context_parts)

        prompt = f"""You are improving an answer based on feedback.

Original Question: {query}

Context:
{context}

Feedback for improvement:
{feedback}

Instructions:
1. Generate an IMPROVED answer addressing the feedback
2. Use inline citations [1], [2], [3]
3. Maintain accuracy and source attribution
4. Address all points in the feedback

Improved Answer:"""

        try:
            response = self.llm.invoke(prompt)
            answer = message_content_to_text(response)
            return self._format_answer(answer, chunks)
        except Exception as e:
            self.log(f"Answer regeneration failed: {str(e)}", level="error")
            raise WriterError(
                message=f"Failed to regenerate answer: {str(e)}",
                details={"feedback": feedback},
            ) from e
