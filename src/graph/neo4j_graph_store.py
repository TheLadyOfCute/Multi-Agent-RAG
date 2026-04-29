"""Neo4j-backed knowledge graph storage for GraphRAG."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set

from src.graph.entity_extractor import Entity
from src.graph.relationship_extractor import Relationship


# Neo4j 图谱的 CRUD
class Neo4jGraphStore:
    """Store and query document knowledge graphs in Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        driver: Any = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "multirag_neo4j")
        if driver is not None:
            self.driver = driver
        else:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if close:
            close()

    def initialize_schema(self) -> None:
        """Create constraints used by the graph store."""
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def is_empty(self) -> bool:
        with self.driver.session() as session:
            row = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()
        return not row or int(row["count"]) == 0

    def counts(self) -> Dict[str, int]:
        with self.driver.session() as session:
            row = session.run(
                "MATCH (e:Entity) "
                "WITH count(e) AS nodes "
                "OPTIONAL MATCH ()-[r:RELATED_TO]->() "
                "RETURN nodes, count(r) AS edges"
            ).single()
        if not row:
            return {"nodes": 0, "edges": 0}
        return {"nodes": int(row["nodes"]), "edges": int(row["edges"])}

    def build_from_chunks(
        self,
        chunks: List[Any],
        chunk_entities: Dict[str, List[Entity]],
        chunk_relationships: Dict[str, List[Relationship]],
    ) -> None:
        """Write extracted entities and relationships to Neo4j."""
        self.initialize_schema()
        print("\nBuilding Neo4j Knowledge Graph...")

        entity_count = 0
        relationship_count = 0
        for chunk_id, entities in chunk_entities.items():
            for entity in entities:
                self.add_entity(entity, chunk_id=chunk_id)
                entity_count += 1

        for chunk_id, relationships in chunk_relationships.items():
            for relationship in relationships:
                self.add_relationship(relationship, chunk_id=chunk_id)
                relationship_count += 1

        counts = self.counts()
        print(
            "Neo4j graph built: "
            f"nodes={counts['nodes']}, edges={counts['edges']}, "
            f"entity_mentions={entity_count}, relationships={relationship_count}"
        )

    def add_entity(self, entity: Entity, chunk_id: Optional[str] = None) -> None:
        """Merge an entity node and attach chunk evidence."""
        name = entity.normalized
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.label = coalesce(e.label, $label),
                    e.text = coalesce(e.text, $text),
                    e.type = 'entity',
                    e.chunk_ids =
                        CASE
                            WHEN $chunk_id IS NULL THEN coalesce(e.chunk_ids, [])
                            WHEN $chunk_id IN coalesce(e.chunk_ids, []) THEN coalesce(e.chunk_ids, [])
                            ELSE coalesce(e.chunk_ids, []) + $chunk_id
                        END
                """,
                name=name,
                label=entity.label,
                text=entity.text,
                chunk_id=chunk_id,
            )

    def add_relationship(
        self,
        relationship: Relationship,
        chunk_id: Optional[str] = None,
    ) -> None:
        """Merge a relationship edge and attach chunk evidence."""
        evidence = None
        evidence_json = None
        if chunk_id:
            evidence = {
                "chunk_id": chunk_id,
                "relation": relationship.relation,
                "confidence": relationship.confidence,
                "extraction_method": getattr(relationship, "extraction_method", "unknown"),
            }
            evidence_json = json.dumps(evidence, ensure_ascii=False, sort_keys=True)

        with self.driver.session() as session:
            session.run(
                """
                MERGE (source:Entity {name: $source})
                ON CREATE SET source.type = 'entity', source.chunk_ids = []
                MERGE (target:Entity {name: $target})
                ON CREATE SET target.type = 'entity', target.chunk_ids = []
                MERGE (source)-[r:RELATED_TO {relation: $relation}]->(target)
                SET r.confidence =
                        CASE
                            WHEN coalesce(r.confidence, 0.0) < $confidence THEN $confidence
                            ELSE r.confidence
                        END,
                    r.extraction_method = coalesce(r.extraction_method, $extraction_method),
                    r.chunk_ids =
                        CASE
                            WHEN $chunk_id IS NULL THEN coalesce(r.chunk_ids, [])
                            WHEN $chunk_id IN coalesce(r.chunk_ids, []) THEN coalesce(r.chunk_ids, [])
                            ELSE coalesce(r.chunk_ids, []) + $chunk_id
                        END,
                    r.evidence_json =
                        CASE
                            WHEN $evidence_json IS NULL THEN coalesce(r.evidence_json, [])
                            WHEN $evidence_json IN coalesce(r.evidence_json, []) THEN coalesce(r.evidence_json, [])
                            ELSE coalesce(r.evidence_json, []) + $evidence_json
                        END
                """,
                source=relationship.source,
                target=relationship.target,
                relation=relationship.relation,
                confidence=relationship.confidence,
                extraction_method=getattr(relationship, "extraction_method", "unknown"),
                chunk_id=chunk_id,
                evidence=evidence,
                evidence_json=evidence_json,
            )

    # 匹配输入实体中已存在于 Neo4j 的实体名称
    def match_entities(self, entities: List[str]) -> List[str]:
        """Return extracted entity names that exist in Neo4j, preserving input order."""
        if not entities:
            return []
        with self.driver.session() as session:
             # 查询输入实体中实际存在于 Entity 节点中的名称
            rows = session.run(
                """
                UNWIND $entities AS candidate
                MATCH (e:Entity {name: candidate})
                RETURN e.name AS name
                """,
                entities=entities,
            )
            found = {row["name"] for row in rows}
        return [entity for entity in entities if entity in found]
    
    # 基于输入实体进行 k-hop 扩展（一级邻跳），返回扩展后的实体集合
    def expand_entities(self, entities: Set[str], k: int = 1) -> Set[str]:
        """Return seed entities plus neighbors up to k hops."""
        if not entities:
            return set()
        with self.driver.session() as session:
            rows = session.run(
                """
                UNWIND $entities AS seed
                MATCH (e:Entity {name: seed})
                OPTIONAL MATCH path = (e)-[:RELATED_TO*1..%d]-(neighbor:Entity)
                RETURN collect(DISTINCT e.name) + collect(DISTINCT neighbor.name) AS names
                """
                % int(k),
                entities=sorted(entities),
            )
            expanded: Set[str] = set()
            for row in rows:
                expanded.update(name for name in row["names"] if name)
        return expanded

    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find directed paths between two entities."""
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH path = (source:Entity {name: $source})-[:RELATED_TO*1..%d]->(target:Entity {name: $target})
                RETURN [node IN nodes(path) | node.name] AS path,
                       [rel IN relationships(path) |
                        {
                          source: startNode(rel).name,
                          target: endNode(rel).name,
                          relation: rel.relation,
                          confidence: coalesce(rel.confidence, 0.0)
                        }] AS relations
                LIMIT 20
                """
                % int(max_length),
                source=source,
                target=target,
            )
            paths = []
            for row in rows:
                path = row["path"]
                relations = row["relations"]
                score = sum(float(rel.get("confidence") or 0.0) for rel in relations)
                if relations:
                    score = score / len(relations)
                paths.append(
                    {
                        "path": path,
                        "relations": relations,
                        "length": max(len(path) - 1, 0),
                        "score": score,
                    }
                )
        return paths

    def collect_evidence_chunk_ids(
        self,
        entities: Set[str],
        paths: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Collect ordered chunk evidence, preferring path edge evidence."""
        if not entities:
            return []
        path_edges = self._path_edges(paths or [])
        with self.driver.session() as session:
            # 给定一批实体名 entities，查这些实体节点，以及这些实体之间的关系边，
            # 最后返回“节点chunk”和“边chunk”。
            rows = session.run(
                """
                MATCH (n:Entity)
                WHERE n.name IN $entities
                OPTIONAL MATCH (n)-[r:RELATED_TO]-(m:Entity)
                WHERE m.name IN $entities
                WITH collect(DISTINCT n.chunk_ids) AS node_lists,
                     collect(DISTINCT CASE
                        WHEN r IS NULL THEN null
                        ELSE {
                            source: startNode(r).name,
                            target: endNode(r).name,
                            chunks: r.chunk_ids
                        }
                     END) AS edge_items
                RETURN
                    reduce(node_chunks = [], item IN node_lists | node_chunks + coalesce(item, [])) AS node_chunk_ids,
                    edge_items AS edge_items
                """,
                entities=sorted(entities),
            )
            row = rows.single()#游标转化为单行数据

        if not row:
            return []

        ordered: List[str] = []  # 保持去重后的有序 chunk 列表
        seen: Set[str] = set()  # 用于去重
        
        def add_many(values: Iterable[str]) -> None:
            """将可迭代中的有效值按出现顺序加入结果列表。"""
            for value in values or []:
                if value and value not in seen:
                    ordered.append(value)
                    seen.add(value)
        
        edge_items = row["edge_items"] or []  # 关联边的 chunk 信息
        if path_edges:
            # 若提供路径边集合，仅优先收集路径上的边
            for item in edge_items:
                if not item:
                    continue
                edge = (item.get("source"), item.get("target"))
                if edge in path_edges:
                    add_many(item.get("chunks"))

        # 再补充所有边上的 chunk（避免遗漏）
        for item in edge_items:
            if item:
                add_many(item.get("chunks"))
        add_many(row["node_chunk_ids"])  # 最后补充节点自身的 chunk
        return ordered

    def _path_edges(self, paths: List[Dict[str, Any]]) -> Set[tuple[str, str]]:
        """从路径列表中提取所有相邻节点对作为边集合，用于构建子图。"""
        edges: Set[tuple[str, str]] = set()
        for path_dict in paths:
            path = path_dict.get("path", [])
            for i in range(len(path) - 1):
                edges.add((path[i], path[i + 1]))
        return edges

    def __repr__(self) -> str:
        try:
            counts = self.counts()
            return f"Neo4jGraphStore(nodes={counts['nodes']}, edges={counts['edges']})"
        except Exception:
            return f"Neo4jGraphStore(uri={self.uri})"
