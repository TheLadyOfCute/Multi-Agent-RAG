"""Embedding generator using DashScope-compatible embeddings API."""

from __future__ import annotations

import time
from typing import Any

from src.config import get_settings
from src.utils.exceptions import AgenticRAGException
from src.utils.logger import setup_logger


class EmbeddingError(AgenticRAGException):
    """Error during embedding generation."""


class EmbeddingGenerator:
    """Generate embeddings using DashScope text-embedding-v4."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        cache_service: Any | None = None,
    ) -> None:
        self.logger = setup_logger("embedder")
        settings = get_settings()

        self.api_key = api_key or settings.dashscope_api_key
        self.base_url = settings.dashscope_base_url
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.batch_size
        self.embedding_dimension = settings.embedding_dimension
        self.cache_service = cache_service

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.logger.info(f"Initialized DashScope embedding client, model: {self.model}")
        except Exception as exc:
            raise EmbeddingError(
                message=f"Failed to initialize DashScope embedding client: {exc}",
                details={"error": str(exc)},
            ) from exc

        self.total_embeddings = 0
        self.total_tokens = 0

     # 批量生成文本嵌入，并优先使用缓存
    def generate(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts, reusing Redis cache when available."""
        if not texts:
            return []

        self.logger.info(f"Generating embeddings for {len(texts)} text(s)")

        embeddings: list[list[float] | None] = [None] * len(texts)
        texts_to_generate: list[str] = []# 保存缓存未命中的文本
        indices_to_generate: list[int] = []# 保存缓存未命中文本在原始列表中的下标

        # 遍历输入文本，优先读取缓存
        for i, text in enumerate(texts):
            cached = self.cache_service.get_embedding(text) if self.cache_service else None
            if cached is not None:
                embeddings[i] = cached
            else:
                texts_to_generate.append(text)
                indices_to_generate.append(i)
        # 按配置的 batch_size 分批生成未命中的嵌入
        for i in range(0, len(texts_to_generate), self.batch_size):
            batch = texts_to_generate[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(texts_to_generate) + self.batch_size - 1) // self.batch_size if texts_to_generate else 0
            self.logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} text(s))")

            try:
                batch_embeddings = self._generate_batch(batch)
                for offset, embedding in enumerate(batch_embeddings):
                    text = batch[offset]
                    target_index = indices_to_generate[i + offset]
                    embeddings[target_index] = embedding
                    if self.cache_service:
                        self.cache_service.set_embedding(text, embedding)
                self.total_embeddings += len(batch)
            except Exception as exc:
                self.logger.error(f"Batch {batch_num} failed: {exc}")
                raise EmbeddingError(
                    message=f"Failed to generate embeddings for batch {batch_num}: {exc}",
                    details={"batch_num": batch_num, "batch_size": len(batch), "error": str(exc)},
                ) from exc

        resolved_embeddings = [embedding for embedding in embeddings if embedding is not None]
        self.logger.info(f"Generated {len(resolved_embeddings)} embeddings (cumulative: {self.total_embeddings})")
        return resolved_embeddings
    
    # 调用模型接口生成一个批次的嵌入
    def _generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                result = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=self.embedding_dimension,
                )
                embeddings = [item.embedding for item in result.data]
                if len(embeddings) != len(texts):
                    raise EmbeddingError(
                        message=f"Expected {len(texts)} embeddings but received {len(embeddings)}",
                        details={"expected": len(texts), "got": len(embeddings)},
                    )
                return embeddings
            except Exception:
                if attempt < max_retries - 1:
                    self.logger.warning("Embedding batch failed, retrying in %ss", retry_delay)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate one query embedding, reusing Redis cache when available."""
        if self.cache_service:
            cached = self.cache_service.get_query_embedding(query)
            if cached is not None:
                return cached

        try:
            result = self.client.embeddings.create(
                input=[query],
                model=self.model,
                dimensions=self.embedding_dimension,
            )
            embedding = result.data[0].embedding
            if self.cache_service:
                self.cache_service.set_query_embedding(query, embedding)
            return embedding
        except Exception as exc:
            raise EmbeddingError(
                message=f"Failed to generate query embedding: {exc}",
                details={"query": query[:100], "error": str(exc)},
            ) from exc

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_embeddings": self.total_embeddings,
            "model": self.model,
            "batch_size": self.batch_size,
            "embedding_dimension": self.get_embedding_dimension(),
        }

    def reset_stats(self) -> None:
        self.total_embeddings = 0
        self.total_tokens = 0
        self.logger.info("Embedding generator stats reset")
