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
            label_row = session.run(
                "CALL db.labels() YIELD label RETURN collect(label) AS labels"
            ).single()
            labels = set(label_row["labels"] if label_row else [])
            if "Entity" not in labels:
                return True
            row = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()
        return not row or int(row["count"]) == 0

    def counts(self) -> Dict[str, int]:
        with self.driver.session() as session:
            label_row = session.run(
                "CALL db.labels() YIELD label RETURN collect(label) AS labels"
            ).single()
            labels = set(label_row["labels"] if label_row else [])
            if "Entity" not in labels:
                return {"nodes": 0, "edges": 0}

            node_row = session.run("MATCH (e:Entity) RETURN count(e) AS nodes").single()
            nodes = int(node_row["nodes"]) if node_row else 0

            type_row = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN collect(relationshipType) AS types"
            ).single()
            relationship_types = set(type_row["types"] if type_row else [])
            if "RELATED_TO" not in relationship_types:
                return {"nodes": nodes, "edges": 0}

            edge_row = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS edges").single()
            edges = int(edge_row["edges"]) if edge_row else 0
        return {"nodes": nodes, "edges": edges}

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
        max_length: int = 3,  # 3跳以内的路径，避免过深遍历导致性能问题
    ) -> List[Dict[str, Any]]:
        """Find directed paths between two entities."""
        with self.driver.session() as session:
            # Cypher 查询：从 source 出发，沿 RELATED_TO 无向边找到 target
            # 使用 *1..max_length 限制路径跳数，避免过深遍历
            max_hops = int(max_length)
            rows = session.run(
                f"""
                MATCH path = (source:Entity {{name: $source}})-[:RELATED_TO*1..{max_hops}]-(target:Entity {{name: $target}})
                RETURN [node IN nodes(path) | node.name] AS path,
                       [rel IN relationships(path) |
                        {{
                          source: startNode(rel).name,
                          target: endNode(rel).name,
                          relation: rel.relation,
                          confidence: coalesce(rel.confidence, 0.0)
                        }}] AS relations
                LIMIT 20
                """,
                source=source,
                target=target,
            )
            paths = []
            for row in rows:
                path = row["path"]
                relations = row["relations"]
                # 路径分数 = 所有关系置信度的平均值
                score = sum(float(rel.get("confidence") or 0.0) for rel in relations)
                if relations:
                    score = score / len(relations)
                paths.append(
                    {
                        "path": path,
                        "relations": relations,
                        "length": max(len(path) - 1, 0),  # 边数 = 节点数 - 1
                        "score": score,
                    }
                )
        return paths

    def collect_evidence_chunk_ids(
        self,
        entities: Set[str],
        paths: Optional[List[Dict[str, Any]]] = None,
        expanded: bool = False,
    ) -> List[str]:
        # Collect ordered chunk evidence.
        # When expanded=False and paths exist, only path-specific edges are
        # queried. When expanded=True, all edges between entities are queried.
        if not entities:
            return []

        ordered: List[str] = []
        seen: Set[str] = set()

        def add_many(values: Iterable[str]) -> None:
            for value in values or []:
                if value and value not in seen:
                    ordered.append(value)
                    seen.add(value)

        path_edge_pairs = self._path_edges(paths or [])

        with self.driver.session() as session:
            # 1. 查询节点的 chunk_ids (始终需要)
            node_rows = session.run(
                """
                MATCH (n:Entity)
                WHERE n.name IN $entities
                RETURN n.name AS name, n.chunk_ids AS chunk_ids
                """,
                entities=sorted(entities),
            )
            for row in node_rows:
                add_many(row["chunk_ids"])

            # 2. 查询边的 chunk_ids
            if not expanded and path_edge_pairs:
                # 实体直接来自路径；仅查询路径中存在的边
                edge_rows = session.run(
                    """
                    UNWIND $edges AS edge
                    MATCH (a:Entity {name: edge[0]})-[r:RELATED_TO]-(b:Entity {name: edge[1]})
                    RETURN DISTINCT r.chunk_ids AS chunks
                    """,
                    edges=[list(e) for e in sorted(path_edge_pairs)],
                )
                for row in edge_rows:
                    add_many(row["chunks"])
            else:
                # 图较稀疏（邻居已扩展）；查询这些实体之间的所有边
                edge_rows = session.run(
                    """
                    MATCH (n:Entity)-[r:RELATED_TO]-(m:Entity)
                    WHERE n.name IN $entities AND m.name IN $entities
                    RETURN DISTINCT r.chunk_ids AS chunks
                    """,
                    entities=sorted(entities),
                )
                for row in edge_rows:
                    add_many(row["chunks"])

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
