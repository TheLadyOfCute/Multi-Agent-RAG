"""
Graph-based retrieval using knowledge-graph paths and entity expansion.
"""

from typing import Any, Dict, List, Set

from src.graph.neo4j_graph_store import Neo4jGraphStore
from src.graph.entity_extractor import EntityExtractor
from src.models.agent_state import Chunk
from src.utils.retrieval_debug import format_ranked_chunk_line


class GraphRetrieval:
    """
    Retrieve chunks using graph paths or entity-neighbor expansion.

    Strategy:
    1. For 2+ matched entities, find paths between entities.
    2. If paths exist, retrieve chunks mentioning path entities.
    3. If no paths exist, fall back to matched entities + one-hop neighbors.
    4. For a single matched entity, retrieve with one-hop neighbors; if that
       returns no chunks, retry with two-hop neighbors.
    """

    def __init__(
        self,
        knowledge_graph: Neo4jGraphStore,
        vector_store,
    ):
        self.kg = knowledge_graph
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 10,
        expand_neighbors: bool = True,
    ) -> List[Chunk]:
        # 提取查询中的实体，并进行规范化处理
        raw_query_entities = self._extract_query_entities(query)
        print(f"\nGraph Query Entities (raw): {raw_query_entities}")

        if not raw_query_entities:
            print(
                "Graph search failed: no_matched_entities | "
                f"raw_entities={raw_query_entities}"
            )
            return []
        # 筛选出查询和图数据库中都存在的实体，确保后续检索基于有效节点
        query_entities = self._filter_entities_in_graph(raw_query_entities)
        print(f"Graph Query Entities (matched_in_graph): {query_entities}")

        if not query_entities:
            print(
                "Graph search failed: no_matched_entities | "
                f"raw_entities={raw_query_entities}"
            )
            return []

        if len(query_entities) == 1:
            return self._search_single_entity_expansion(query_entities[0], top_k)

        #对于多个实体的查询，尝试在它们之间找到连接路径，并基于路径相关实体进行检索
        all_paths = []
        for i, source in enumerate(query_entities):
            for target in query_entities[i + 1:]:
                #双向查询，因为图数据是有向的，路径可能存在于任一方向
                all_paths.extend(self._find_paths(source, target, max_length=3))#3跳以内的路径
        #对找到的路径进行排序，优先考虑更短、更具体、置信度更高的路径
        ranked_paths = self._rank_paths(all_paths)
        print(f"\nGraph paths found: {len(ranked_paths)}")

        #打印前 5 条路径的详细信息，便于调试路径质量和相关实体覆盖情况
        for idx, path_dict in enumerate(ranked_paths[:5], start=1):
            path = path_dict.get("path", [])
            score = path_dict.get("score", 0)
            relations = path_dict.get("relations", [])
            relation_labels = [rel.get("relation", "related_to") for rel in relations]
            print(
                f"  {idx}. {' -> '.join(path)} | "
                f"relations={relation_labels} | score={score:.4f}"
            )

        if not ranked_paths:
            print("No paths found in graph; falling back to entity-neighbor expansion")
            return self._search_multi_entity_neighbor_fallback(set(query_entities), top_k)

        # 选取前 5 条评分最高的路径进行实体提取
        paths = ranked_paths[:5]
        # 从路径中收集所有独特的实体，作为后续检索的目标实体集合
        path_entities = self._collect_path_entities(paths)

        # 先用路径实体直接检索，不开邻居扩展
        print(
            f"Graph search targeting {len(path_entities)} entities: "
            f"{list(path_entities)[:5]}..."
        )
        chunks = self._retrieve_chunks_by_entities(
            path_entities,
            top_k=top_k * 2,
            paths=paths,
        )

        # 图稀疏时（chunks 不足），才开启邻居扩展补充证据
        if expand_neighbors and len(chunks) < top_k:
            expanded_entities = self._expand_with_neighbors(path_entities, k=1)
            print(
                f"Graph sparse (chunks={len(chunks)} < top_k={top_k}), "
                f"expanding neighbors: {len(path_entities)} -> {len(expanded_entities)} entities"
            )
            chunks = self._retrieve_chunks_by_entities(
                expanded_entities,
                top_k=top_k * 2,
                paths=paths,
                expanded=True,
            )
            path_entities = expanded_entities

        # 根据路径相关性对检索到的分块进行重排序
        ranked_chunks = self._rank_by_path_relevance(chunks, paths, path_entities)

        # 为检索到的分块标记状态和目标实体信息
        self._tag_graph_chunks(
            ranked_chunks,
            status="path_search",
            target_entities=path_entities,
        )
        return ranked_chunks[:top_k]

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract and normalize query entities using the same extractor as ingestion."""
        extractor = EntityExtractor()
        entities = extractor.extract(query)
        normalized = []
        seen = set()
        # 遍历提取出的实体对象
        for entity in entities:
             # 读取实体的 normalized 字段，并统一转为小写字符串
            value = str(getattr(entity, "normalized", "")).strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _filter_entities_in_graph(self, entities: List[str]) -> List[str]:
        #过滤出存在于知识图谱中的实体，确保后续检索基于有效节点
        return self.kg.match_entities(entities)

    def _search_single_entity_expansion(
        self,
        seed: str,
        top_k: int,
    ) -> List[Chunk]:
        """Retrieve with one-hop neighbors, then two-hop neighbors if empty."""
        target_entities = self._expand_with_neighbors({seed}, k=1)

        print("\nGraph single-entity expansion:")
        print(f"  seed={seed}")
        print("  hop=1")
        print(f"  target_entities={sorted(target_entities)}")
        #用双倍的 top_k 去检索 chunk，给排序留出余量
        #
        chunks = self._retrieve_chunks_by_entities(target_entities, top_k=top_k * 2)
        print(f"  chunks={len(chunks)}")
        expansion_hops = 1

        if not chunks and seed:#如果一跳邻居没有检索到相关 chunk，尝试两跳邻居扩展
            target_entities = self._expand_with_neighbors({seed}, k=2)
            print("\nGraph single-entity expansion:")
            print(f"  seed={seed}")
            print("  hop=2")
            print(f"  target_entities={sorted(target_entities)}")
            #用双倍的 top_k 去检索 chunk，给排序留出余量
            chunks = self._retrieve_chunks_by_entities(target_entities, top_k=top_k * 2)
            print(f"  chunks={len(chunks)}")
            expansion_hops = 2
        # 按图路径相关性重新打分
        ranked_chunks = self._rank_by_path_relevance(chunks, [], target_entities)
        #附加图检索相关的元数据标签，便于后续分析与调试
        self._tag_graph_chunks(
            ranked_chunks,
            status="single_entity_expansion",
            target_entities=target_entities,
            expansion_hops=expansion_hops,
        )
        return ranked_chunks[:top_k]

    def _search_multi_entity_neighbor_fallback(
        self,
        matched_entities: Set[str],
        top_k: int,
    ) -> List[Chunk]:
        """Fallback for matched multi-entity queries with no connecting paths."""
        target_entities = self._expand_with_neighbors(matched_entities, k=1)
        print(
            "Graph multi-entity neighbor fallback: "
            f"entities={sorted(matched_entities)}, "
            f"target_entities={sorted(target_entities)}"
        )
        chunks = self._retrieve_chunks_by_entities(target_entities, top_k=top_k * 2)
        ranked_chunks = self._rank_by_path_relevance(chunks, [], target_entities)
        self._tag_graph_chunks(
            ranked_chunks,
            status="multi_entity_neighbor_fallback",
            target_entities=target_entities,
            expansion_hops=1,
        )
        return ranked_chunks[:top_k]

    def _tag_graph_chunks(
        self,
        chunks: List[Chunk],
        status: str,
        target_entities: Set[str],
        expansion_hops: int = None,
    ) -> None:
        """Attach graph retrieval trace metadata to returned chunks."""
        for chunk in chunks:
            chunk.metadata["source"] = "graph"
            chunk.metadata["retriever"] = "graph"
            chunk.metadata["graph_status"] = status
            chunk.metadata["graph_target_entities"] = sorted(target_entities)
            if expansion_hops is not None:
                chunk.metadata["graph_expansion_hops"] = expansion_hops

    def _collect_path_entities(self, paths: List[Dict]) -> Set[str]:
        """Collect all unique entities from path dictionaries."""
        entities = set()
        for path_dict in paths:
            #获取每个路径上的实体
            entities.update(path_dict.get("path", []))
        return entities

    def _find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3,
    ) -> List[Dict]:
        """Find paths between two graph entities."""
        return self.kg.find_paths(source, target, max_length=max_length)

    def _rank_paths(self, paths: List[Dict]) -> List[Dict]:
        """Rank candidate paths by length, confidence, and relation specificity."""
        scored_paths = []
        for path_dict in paths:
            score = 0.0
            # 短路径优先：长度越短分数越高
            score += (1.0 / path_dict["length"]) * 2.0

            relations = path_dict["relations"]
            if relations:
                # 关系置信度的平均值加权
                # 默认创建图数据库设置的confidence：
                # 共现关系 0.5
                # 模式匹配 0.8
                # 句法分析 0.7
                avg_confidence = sum(r.get("confidence", 0.5) for r in relations) / len(relations)
                score += avg_confidence

            # 排除笼统关系，偏好更具体的关系类型
            specific_relations = [r for r in relations if r.get("relation") != "related_to"]
            if specific_relations:
                # 具体关系数量越多分数越高
                score += len(specific_relations) * 0.5

            # 写回得分用于排序
            path_dict["score"] = score
            scored_paths.append(path_dict)

        return sorted(scored_paths, key=lambda item: item["score"], reverse=True)

    def _expand_with_neighbors(self, entities: Set[str], k: int = 1) -> Set[str]:
        """Expand entity set with predecessors and successors up to k hops."""
        #返回一级邻跳
        return self.kg.expand_entities(entities, k=k)

    def _retrieve_chunks_by_entities(
        self,
        entities: Set[str],
        top_k: int = 20,
        paths: List[Dict] = None,
        expanded: bool = False,
    ) -> List[Chunk]:
        #收集所有实体的相关（包括边） chunk_id，路径信息可用于后续排序，但不影响检索覆盖
        chunk_ids = self._collect_evidence_chunk_ids(entities, paths=paths, expanded=expanded)
        print("\nGraph evidence chunks:")
        print(f"  target_entities={sorted(entities)}")
        print(f"  path_count={len(paths or [])}")
        print(f"  chunk_ids={chunk_ids}")
        if not chunk_ids:
            print("  fetched_chunks=0")
            return []

        if not hasattr(self.vector_store, "get_chunks_by_ids"):
            print("Graph search failed: vector store cannot fetch chunks by id")
            return []
        
        #根据 chunk_id 从向量数据库中获取对应的 chunks，并附加图检索相关的元数据标签
        results = self.vector_store.get_chunks_by_ids(chunk_ids[:top_k])

        # 将向量库返回结果转换为统一的 Chunk 对象
        chunks = []
        for result in results:
            # 复制元数据，避免直接修改原始结果中的 metadata
            metadata = dict(result.get("metadata", {}) or {})
            chunk = Chunk(
                text=result["text"],
                doc_id="unknown",
                chunk_id=result["chunk_id"],
                score=result["score"],
                metadata={
                    # 图检索默认字段，可被后续 **metadata 覆盖
                    "filename": metadata.get("filename", "unknown"),
                    "retrieval_method": "graph",
                    "source": "graph",
                    "retriever": "graph",
                    **metadata,
                },
            )
            chunks.append(chunk)

        # 打印检索统计与排序展示，便于调试图证据命中情况
        print(f"  fetched_chunks={len(chunks)}")
        for rank, chunk in enumerate(chunks, start=1):
            print(f"  {format_ranked_chunk_line(rank, chunk)}")

        return chunks

    def _collect_evidence_chunk_ids(
        self,
        entities: Set[str],
        paths: List[Dict] = None,
        expanded: bool = False,
    ) -> List[str]:
        """Collect ordered chunk evidence from matching nodes and edges."""
        return self.kg.collect_evidence_chunk_ids(entities, paths=paths, expanded=expanded)

    def _rank_by_path_relevance(
        self,
        chunks: List[Chunk],
        paths: List[Dict],
        path_entities: Set[str],
    ) -> List[Chunk]:
        """Re-rank chunks by path/entity coverage."""
        scored_chunks = []

        for chunk in chunks:
            text_lower = chunk.text.lower()
            # 统计与路径实体匹配的数量
            entity_count = sum(1 for entity in path_entities if entity in text_lower)

            # 初始化路径相关性奖励分
            path_bonus = 0.0
            # 只取前 3 条路径参与路径奖励计算
            for path_dict in paths[:3]:
                # 获取路径中的实体列表
                path = path_dict.get("path", [])

                # 获取当前路径本身的分数，之前计算过的
                """
                neo4j_graph_store.py:253:
                find_paths 从 Neo4j 查询路径时，计算一个初始分数：所有关系 confidence 的平均值。
                graph_retrieval.py:269:
                _rank_paths 用更复杂的公式覆盖了初始 score:
                score = (1.0 / path_length) * 2.0 + avg_confidence + specific_relation_bonus
                """
                path_score = path_dict.get("score", 0)

                # 统计当前文档块中命中的路径实体数量
                path_mentions = sum(1 for entity in path if entity in text_lower)

                # 至少命中 2 个路径实体，才认为与该路径相关并计入奖励
                if path_mentions >= 2:
                    path_bonus += path_score * 0.5

            # 在原始分数基础上叠加实体覆盖奖励和路径奖励
            chunk.score = (chunk.score or 0.0) + entity_count * 0.3 + path_bonus

            # 保存重新打分后的文档块
            scored_chunks.append(chunk)

        return sorted(scored_chunks, key=lambda c: c.score, reverse=True)
