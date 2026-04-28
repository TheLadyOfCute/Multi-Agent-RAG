"""Redis-backed cache helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.config import get_settings
from src.utils.logger import setup_logger

"""
Redis key = 类型前缀 + SHA256 hash
Redis value = JSON 字符串
过期策略 = setex + TTL, 默认 1 小时
取出流程 = get -> json.loads -> hit/miss 统计
删除流程 = scan_iter(prefix*) -> delete
"""


class RedisCacheService:
    """Centralized Redis cache for embeddings, answers, and cache stats."""

    # 文档嵌入缓存 Key 前缀
    EMBEDDING_PREFIX = "cache:embedding:"

    # 查询嵌入缓存 Key 前缀
    QUERY_EMBEDDING_PREFIX = "cache:query_embedding:"

    # 回答缓存 Key 前缀
    ANSWER_PREFIX = "cache:answer:"

    # 缓存命中次数统计 Key
    HIT_KEY = "cache:stats:hits"

    # 缓存未命中次数统计 Key
    MISS_KEY = "cache:stats:misses"

    # 初始化 Redis 缓存服务，并完成可用性检测
    def __init__(
        self,
        *,
        client: Any | None = None,
        redis_url: str | None = None,
        enabled: bool | None = None,
        ttl: int | None = None,
    ) -> None:
        # 创建缓存日志器并读取应用配置
        self.logger = setup_logger("redis_cache") or logging.getLogger("redis_cache")
        settings = get_settings()

        # 支持通过参数覆盖默认配置
        self.enabled = settings.cache_enabled if enabled is None else enabled
        self.ttl = settings.cache_ttl if ttl is None else ttl
        self.redis_url = redis_url or settings.redis_url

        # 初始化客户端和可用状态
        self.client = None
        self.available = False

        # 配置关闭缓存时直接退出
        if not self.enabled:
            self.logger.info("Redis cache disabled via configuration")
            return

        # 测试或外部注入场景下直接使用传入客户端
        if client is not None:
            self.client = client
            self.available = True
            return

        try:
            # 延迟导入 Redis，避免缺少依赖时影响模块加载
            from redis import Redis
        except ModuleNotFoundError as exc:
            # Redis 包未安装时记录警告并保持不可用状态
            self.logger.warning("Redis cache unavailable: redis package is not installed (%s)", exc)
            return

        try:
            # 创建 Redis 客户端并设置连接超时
            self.client = Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            # 通过 ping 验证 Redis 是否可连接
            self.client.ping()
            self.available = True

            # 记录连接成功信息
            self.logger.info("Redis cache connected: %s", self.redis_url)
        except Exception as exc:
            # Redis 连接失败时记录原因，并清空客户端
            self.logger.warning("Redis cache unavailable at %s: %s", self.redis_url, exc)
            self.client = None

    # 获取普通文档文本的嵌入缓存
    def get_embedding(self, text: str) -> list[float] | None:
        # 复用统一的向量缓存读取逻辑
        return self._get_vector(self.EMBEDDING_PREFIX, text)

    # 写入普通文档文本的嵌入缓存
    def set_embedding(self, text: str, embedding: list[float]) -> None:
        # 复用统一的向量缓存写入逻辑
        self._set_vector(self.EMBEDDING_PREFIX, text, embedding)

    # 获取用户查询文本的嵌入缓存
    def get_query_embedding(self, query: str) -> list[float] | None:
        # 查询嵌入使用独立前缀，避免和文档嵌入混用
        return self._get_vector(self.QUERY_EMBEDDING_PREFIX, query)

    # 写入用户查询文本的嵌入缓存
    def set_query_embedding(self, query: str, embedding: list[float]) -> None:
        # 查询嵌入单独缓存，便于重复问题加速
        self._set_vector(self.QUERY_EMBEDDING_PREFIX, query, embedding)

    # 根据查询和知识库状态读取回答缓存
    def get_answer(self, query: str, knowledge_state_token: str) -> dict[str, Any] | None:
        # Redis 不可用时直接视为缓存未命中
        if not self.available:
            return None

        # 通过查询文本和知识状态生成缓存 Key
        payload = self._safe_get(self._answer_key(query, knowledge_state_token))

        # 没有缓存内容时记录 miss
        if payload is None:
            self._record_miss()
            return None

        try:
            # 将缓存中的 JSON 字符串还原为字典
            value = json.loads(payload)
            self._record_hit()
            return value
        except json.JSONDecodeError:
            # 缓存内容损坏时按未命中处理
            self.logger.warning("Invalid cached answer payload; treating as miss")
            self._record_miss()
            return None

    # 写入查询回答缓存
    def set_answer(self, query: str, knowledge_state_token: str, payload: dict[str, Any]) -> None:
        # Redis 不可用时不执行写入
        if not self.available:
            return

        # 序列化回答内容，并按 TTL 写入缓存
        self._safe_set(self._answer_key(query, knowledge_state_token), json.dumps(payload, ensure_ascii=False))

    # 清空所有回答缓存
    def clear_answer_cache(self) -> int:
        # 仅删除回答缓存，不影响嵌入缓存
        return self._delete_by_prefix(self.ANSWER_PREFIX)

    # 获取缓存命中率、命中次数、未命中次数和缓存规模
    def get_stats(self) -> dict[str, Any]:
        # Redis 不可用时返回默认空统计
        if not self.available:
            return {
                "cache_hit_rate": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_size": 0,
            }

        # 读取命中和未命中计数
        hits = self._safe_int(self._safe_get(self.HIT_KEY))
        misses = self._safe_int(self._safe_get(self.MISS_KEY))
        total = hits + misses

        # 统计三类缓存 Key 的总数量
        cache_size = (
            self._count_by_prefix(self.EMBEDDING_PREFIX)
            + self._count_by_prefix(self.QUERY_EMBEDDING_PREFIX)
            + self._count_by_prefix(self.ANSWER_PREFIX)
        )

        # 汇总返回缓存统计结果
        return {
            "cache_hit_rate": hits / total if total else 0.0,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_size": cache_size,
        }

    # 根据文档列表生成知识库状态标识
    @staticmethod
    def build_knowledge_state_token(documents: list[dict[str, Any]]) -> str:
        # 没有文档时使用固定 token，方便缓存隔离
        if not documents:
            return "empty"

        # 只保留会影响知识状态的字段，并按文档名排序保证稳定性
        normalized = [
            {
                "name": doc.get("name", ""),
                "chunks": doc.get("chunks", 0),
                "uploaded_at": doc.get("uploaded_at", ""),
            }
            for doc in sorted(documents, key=lambda item: str(item.get("name", "")))
        ]

        # 生成稳定 JSON，再计算哈希作为知识库版本
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()#SHA-256哈希值id

    # 规范化查询文本，减少等价查询的缓存重复
    @staticmethod
    def normalize_query(query: str) -> str:
        # 合并多余空白字符
        return " ".join(query.split())

    # 从 Redis 中读取向量缓存
    def _get_vector(self, prefix: str, text: str) -> list[float] | None:
        # Redis 不可用时不读取缓存
        if not self.available:
            return None

        # 根据前缀和文本哈希定位缓存
        payload = self._safe_get(self._vector_key(prefix, text))

        # 无缓存时记录未命中
        if payload is None:
            self._record_miss()
            return None

        try:
            # 反序列化缓存中的向量
            value = json.loads(payload)
            self._record_hit()
            return value
        except json.JSONDecodeError:
            # 缓存数据异常时按未命中处理
            self.logger.warning("Invalid cached vector payload; treating as miss")
            self._record_miss()
            return None

    # 将向量写入 Redis 缓存
    def _set_vector(self, prefix: str, text: str, embedding: list[float]) -> None:
        # Redis 不可用时不写入缓存
        if not self.available:
            return

        # 将向量转成 JSON 字符串后写入
        self._safe_set(self._vector_key(prefix, text), json.dumps(embedding))

    # 生成向量缓存 Key
    def _vector_key(self, prefix: str, text: str) -> str:
        # 使用文本哈希避免 Key 过长或包含特殊字符
        return prefix + self._hash_text(text)

    # 生成回答缓存 Key
    def _answer_key(self, query: str, knowledge_state_token: str) -> str:
        # 先规范化查询文本，提升缓存复用率
        normalized = self.normalize_query(query)

        # 回答缓存同时绑定知识库状态，避免文档变化后误用旧答案
        return self.ANSWER_PREFIX + knowledge_state_token + ":" + self._hash_text(normalized)

    # 计算文本的 SHA-256 哈希
    @staticmethod
    def _hash_text(value: str) -> str:
        # 使用 UTF-8 编码保证中英文文本都能稳定计算
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    # 记录一次缓存命中
    def _record_hit(self) -> None:
        # 命中计数独立存储，便于统计缓存效果
        self._safe_incr(self.HIT_KEY)

    # 记录一次缓存未命中
    def _record_miss(self) -> None:
        # 未命中计数独立存储，便于计算命中率
        self._safe_incr(self.MISS_KEY)

    # 删除指定前缀下的所有缓存
    def _delete_by_prefix(self, prefix: str) -> int:
        # Redis 不可用时没有可删除内容
        if not self.available:
            return 0

        # 扫描指定前缀的所有 Key
        keys = list(self.client.scan_iter(match=prefix + "*"))

        # 没有匹配 Key 时返回 0
        if not keys:
            return 0

        # 批量删除匹配到的 Key
        return int(self.client.delete(*keys))

    # 统计指定前缀下的缓存数量
    def _count_by_prefix(self, prefix: str) -> int:
        # Redis 不可用时缓存数量为 0
        if not self.available:
            return 0

        # 使用 scan_iter 避免一次性阻塞扫描
        return sum(1 for _ in self.client.scan_iter(match=prefix + "*"))

    # 安全读取 Redis Key
    def _safe_get(self, key: str) -> str | None:
        try:
            # 读取失败不抛出到上层，避免缓存影响主流程
            return self.client.get(key)
        except Exception as exc:
            self.logger.warning("Redis get failed for %s: %s", key, exc)
            return None

    # 安全写入 Redis Key
    def _safe_set(self, key: str, value: str) -> None:
        try:
            # 使用 setex 写入并设置过期时间
            self.client.setex(key, self.ttl, value)
        except Exception as exc:
            self.logger.warning("Redis set failed for %s: %s", key, exc)

    # 安全递增 Redis 计数器
    def _safe_incr(self, key: str) -> None:
        # Redis 不可用时跳过统计
        if not self.available:
            return

        try:
            # 递增指定统计 Key
            self.client.incr(key)
        except Exception as exc:
            self.logger.warning("Redis incr failed for %s: %s", key, exc)

    # 将任意值安全转换为整数
    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            # 空值按 0 处理
            return int(value or 0)
        except (TypeError, ValueError):
            # 非法数字格式也按 0 处理
            return 0
