"""
Performance Tracker - Week 5 Day 6
Track latency, cache hits, and agent execution times.
"""

import time
from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path


class PerformanceTracker:
    """Track system performance metrics."""

    def __init__(self):
        # 初始化指标列表，用于存储每次查询的性能数据
        self.metrics = []

        # 记录当前会话的开始时间
        self.session_start = datetime.now()

    def track_query(
        self,
        query: str,
        latency: float,
        chunks_retrieved: int,
        strategy: str,
        iterations: int,
        cache_hit: bool = False
    ) -> None:
        """Record query performance."""

        # 构建单次查询的性能指标数据
        metric = {
            # 记录当前查询的时间戳
            'timestamp': datetime.now().isoformat(),

            # 记录查询文本长度
            'query_length': len(query),

            # 将查询延迟从秒转换为毫秒
            'latency_ms': latency * 1000,

            # 记录本次查询检索到的文本块数量
            'chunks_retrieved': chunks_retrieved,

            # 记录本次查询使用的检索或执行策略
            'strategy': strategy,

            # 记录本次查询的迭代次数
            'iterations': iterations,

            # 记录本次查询是否命中缓存
            'cache_hit': cache_hit
        }

        # 将本次查询指标追加到指标列表中
        self.metrics.append(metric)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""

        # 如果没有任何指标数据，则返回空统计结果
        if not self.metrics:
            return {}

        # 提取所有查询的延迟数据
        latencies = [m['latency_ms'] for m in self.metrics]

        # 统计缓存命中的查询数量
        cache_hits = sum(1 for m in self.metrics if m.get('cache_hit'))

        # 返回聚合后的性能统计信息
        return {
            # 查询总次数
            'total_queries': len(self.metrics),

            # 平均查询延迟
            'avg_latency_ms': sum(latencies) / len(latencies),

            # 最小查询延迟
            'min_latency_ms': min(latencies),

            # 最大查询延迟
            'max_latency_ms': max(latencies),

            # 缓存命中率
            'cache_hit_rate': cache_hits / len(self.metrics) if self.metrics else 0,

            # 平均检索文本块数量
            'avg_chunks': sum(m['chunks_retrieved'] for m in self.metrics) / len(self.metrics),

            # 当前会话持续时间，单位为分钟
            'session_duration_min': (datetime.now() - self.session_start).total_seconds() / 60
        }

    def save_metrics(self, filepath: str = "data/metrics.json") -> None:
        """Save metrics to file."""

        # 确保指标文件所在目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # 将性能指标写入 JSON 文件
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)