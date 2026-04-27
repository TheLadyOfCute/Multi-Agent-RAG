"""
Graph-based retrieval using knowledge-graph paths and entity expansion.
"""

from typing import Dict, List, Set
import networkx as nx

from src.graph.entity_extractor import EntityExtractor
from src.graph.graph_builder import KnowledgeGraph
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
        knowledge_graph: KnowledgeGraph,
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
        """Search the knowledge graph using query entities from EntityExtractor."""
        raw_query_entities = self._extract_query_entities(query)
        print(f"\nGraph Query Entities (raw): {raw_query_entities}")

        if not raw_query_entities:
            print(
                "Graph search failed: no_matched_entities | "
                f"raw_entities={raw_query_entities}"
            )
            return []

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

        all_paths = []
        for i, source in enumerate(query_entities):
            for target in query_entities[i + 1:]:
                all_paths.extend(self._find_paths(source, target, max_length=3))

        ranked_paths = self._rank_paths(all_paths)
        print(
            "\nGraph paths found: "
            f"total={len(all_paths)}, returned={min(5, len(ranked_paths))}"
        )
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

        paths = ranked_paths[:5]
        path_entities = self._collect_path_entities(paths)
        if expand_neighbors:
            path_entities = self._expand_with_neighbors(path_entities, k=1)

        print(
            f"Graph search targeting {len(path_entities)} entities: "
            f"{list(path_entities)[:5]}..."
        )

        chunks = self._retrieve_chunks_by_entities(
            path_entities,
            top_k=top_k * 2,
            paths=paths,
        )
        ranked_chunks = self._rank_by_path_relevance(chunks, paths, path_entities)
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
        for entity in entities:
            value = str(getattr(entity, "normalized", "")).strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _filter_entities_in_graph(self, entities: List[str]) -> List[str]:
        """Keep only entities that exist as nodes in the knowledge graph."""
        if hasattr(self.kg, "match_entities"):
            return self.kg.match_entities(entities)
        return [entity for entity in entities if entity in self.kg.graph]

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
        chunks = self._retrieve_chunks_by_entities(target_entities, top_k=top_k * 2)
        print(f"  chunks={len(chunks)}")
        expansion_hops = 1

        if not chunks and seed:
            target_entities = self._expand_with_neighbors({seed}, k=2)
            print("\nGraph single-entity expansion:")
            print(f"  seed={seed}")
            print("  hop=2")
            print(f"  target_entities={sorted(target_entities)}")
            chunks = self._retrieve_chunks_by_entities(target_entities, top_k=top_k * 2)
            print(f"  chunks={len(chunks)}")
            expansion_hops = 2

        ranked_chunks = self._rank_by_path_relevance(chunks, [], target_entities)
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
            entities.update(path_dict.get("path", []))
        return entities

    def _find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3,
    ) -> List[Dict]:
        """Find paths between two graph entities."""
        if hasattr(self.kg, "find_paths"):
            return self.kg.find_paths(source, target, max_length=max_length)

        if source not in self.kg.graph or target not in self.kg.graph:
            return []

        try:
            paths = list(nx.all_simple_paths(self.kg.graph, source, target, cutoff=max_length))
        except nx.NetworkXNoPath:
            return []

        path_dicts = []
        for path in paths:
            relations = []
            for i in range(len(path) - 1):
                edge_data = self.kg.graph[path[i]][path[i + 1]]
                relations.append(
                    {
                        "from": path[i],
                        "to": path[i + 1],
                        "relation": edge_data.get("relation", "related_to"),
                        "confidence": edge_data.get("confidence", 0.5),
                    }
                )
            path_dicts.append(
                {
                    "path": path,
                    "length": len(path),
                    "relations": relations,
                }
            )
        return path_dicts

    def _rank_paths(self, paths: List[Dict]) -> List[Dict]:
        """Rank candidate paths by length, confidence, and relation specificity."""
        scored_paths = []
        for path_dict in paths:
            score = 0.0
            score += (1.0 / path_dict["length"]) * 2.0

            relations = path_dict["relations"]
            if relations:
                avg_confidence = sum(r.get("confidence", 0.5) for r in relations) / len(relations)
                score += avg_confidence

            specific_relations = [r for r in relations if r.get("relation") != "related_to"]
            if specific_relations:
                score += len(specific_relations) * 0.5

            path_dict["score"] = score
            scored_paths.append(path_dict)

        return sorted(scored_paths, key=lambda item: item["score"], reverse=True)

    def _expand_with_neighbors(self, entities: Set[str], k: int = 1) -> Set[str]:
        """Expand entity set with predecessors and successors up to k hops."""
        if hasattr(self.kg, "expand_entities"):
            return self.kg.expand_entities(entities, k=k)
        expanded = set(entities)
        frontier = set(entities)
        for _ in range(k):
            next_frontier = set()
            for entity in frontier:
                if entity not in self.kg.graph:
                    continue
                next_frontier.update(self.kg.graph.successors(entity))
                next_frontier.update(self.kg.graph.predecessors(entity))
            next_frontier -= expanded
            expanded.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return expanded

    def _retrieve_chunks_by_entities(
        self,
        entities: Set[str],
        top_k: int = 20,
        paths: List[Dict] = None,
    ) -> List[Chunk]:
        """Retrieve chunks attached to graph entity nodes and relationship edges."""
        chunk_ids = self._collect_evidence_chunk_ids(entities, paths=paths)
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

        results = self.vector_store.get_chunks_by_ids(chunk_ids[:top_k])

        chunks = []
        for result in results:
            metadata = dict(result.get("metadata", {}) or {})
            if result.get("child_chunk_id"):
                metadata["child_chunk_id"] = result["child_chunk_id"]
            elif result.get("chunk_type") == "child":
                metadata["child_chunk_id"] = result["chunk_id"]
            chunk = Chunk(
                text=result["text"],
                doc_id="unknown",
                chunk_id=result["chunk_id"],
                score=result["score"],
                metadata={
                    "filename": metadata.get("filename", "unknown"),
                    "chunk_type": result.get("chunk_type", "child"),
                    "retrieval_method": "graph",
                    "source": "graph",
                    "retriever": "graph",
                    **metadata,
                },
            )
            chunks.append(chunk)

        print(f"  fetched_chunks={len(chunks)}")
        for rank, chunk in enumerate(chunks, start=1):
            print(f"  {format_ranked_chunk_line(rank, chunk)}")

        return chunks

    def _collect_evidence_chunk_ids(
        self,
        entities: Set[str],
        paths: List[Dict] = None,
    ) -> List[str]:
        """Collect ordered child chunk evidence from matching nodes and edges."""
        if hasattr(self.kg, "collect_evidence_chunk_ids"):
            return self.kg.collect_evidence_chunk_ids(entities, paths=paths)

        ordered_ids = []
        seen = set()

        def add_ids(chunk_ids):
            for chunk_id in chunk_ids or []:
                if chunk_id and chunk_id not in seen:
                    ordered_ids.append(chunk_id)
                    seen.add(chunk_id)

        for entity in sorted(entities):
            if entity in self.kg.graph:
                add_ids(self.kg.graph.nodes[entity].get("chunk_ids", []))

        for source, target, data in self.kg.graph.edges(data=True):
            if source in entities and target in entities:
                add_ids(data.get("chunk_ids", []))
                add_ids(item.get("chunk_id") for item in data.get("evidence", []))

        return ordered_ids

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
            entity_count = sum(1 for entity in path_entities if entity in text_lower)

            path_bonus = 0.0
            for path_dict in paths[:3]:
                path = path_dict.get("path", [])
                path_score = path_dict.get("score", 0)
                path_mentions = sum(1 for entity in path if entity in text_lower)
                if path_mentions >= 2:
                    path_bonus += path_score * 0.5

            chunk.score = (chunk.score or 0.0) + entity_count * 0.3 + path_bonus
            scored_chunks.append(chunk)

        return sorted(scored_chunks, key=lambda c: c.score, reverse=True)
