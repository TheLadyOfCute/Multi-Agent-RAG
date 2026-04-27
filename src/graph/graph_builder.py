"""
Knowledge Graph Builder - Week 9 Day 3
Build NetworkX graph from entities and relationships.
"""

import networkx as nx
from typing import List, Dict, Set, Tuple
from pathlib import Path
import pickle

from src.graph.entity_extractor import Entity
from src.graph.relationship_extractor import Relationship


class KnowledgeGraph:
    """
    Knowledge graph built from document entities and relationships.
    
    Structure:
    - Nodes: Entities (with labels and metadata)
    - Edges: Relationships (with relation types and confidence)
    """
    
    def __init__(self):
        """Initialize empty directed graph."""
        self.graph = nx.DiGraph()
        self.entity_count = 0
        self.relationship_count = 0
    
    def add_entity(self, entity: Entity, chunk_id: str = None) -> None:
        """
        Add entity as node.
        
        Args:
            entity: Entity object to add
        """
        node_id = entity.normalized
        
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                label=entity.label,
                text=entity.text,
                type='entity',
                chunk_ids=[]
            )
            self.entity_count += 1
        else:
            self.graph.nodes[node_id].setdefault("chunk_ids", [])

        if chunk_id and chunk_id not in self.graph.nodes[node_id]["chunk_ids"]:
            self.graph.nodes[node_id]["chunk_ids"].append(chunk_id)
    
    def add_relationship(self, relationship: Relationship, chunk_id: str = None) -> None:
        """
        Add relationship as edge.
        
        Args:
            relationship: Relationship object to add
        """
        # Ensure nodes exist
        if relationship.source not in self.graph:
            self.graph.add_node(relationship.source, type='entity', chunk_ids=[])
        
        if relationship.target not in self.graph:
            self.graph.add_node(relationship.target, type='entity', chunk_ids=[])

        evidence = None
        if chunk_id:
            evidence = {
                "chunk_id": chunk_id,
                "relation": relationship.relation,
                "confidence": relationship.confidence,
                "extraction_method": getattr(relationship, "extraction_method", "unknown"),
            }
        
        # Add edge (or update if exists)
        if self.graph.has_edge(relationship.source, relationship.target):
            # Update with higher confidence if better
            existing = self.graph[relationship.source][relationship.target]
            existing.setdefault("chunk_ids", [])
            existing.setdefault("evidence", [])
            if chunk_id and chunk_id not in existing["chunk_ids"]:
                existing["chunk_ids"].append(chunk_id)
            if evidence and evidence not in existing["evidence"]:
                existing["evidence"].append(evidence)
            if relationship.confidence > existing.get('confidence', 0):
                self.graph[relationship.source][relationship.target].update({
                    'relation': relationship.relation,
                    'confidence': relationship.confidence
                })
        else:
            chunk_ids = [chunk_id] if chunk_id else []
            evidence_items = [evidence] if evidence else []
            self.graph.add_edge(
                relationship.source,
                relationship.target,
                relation=relationship.relation,
                confidence=relationship.confidence,
                extraction_method=getattr(relationship, "extraction_method", "unknown"),
                chunk_ids=chunk_ids,
                evidence=evidence_items
            )
            self.relationship_count += 1
    
    def build_from_chunks(
        self,
        chunks: List[any],
        chunk_entities: Dict[str, List[Entity]],
        chunk_relationships: Dict[str, List[Relationship]]
    ) -> None:
        """
        Build graph from chunks with extracted entities and relationships.
        
        Args:
            chunks: List of Chunk objects
            chunk_entities: Dict mapping chunk_id to entities
            chunk_relationships: Dict mapping chunk_id to relationships
        """
        print("\n🔨 Building Knowledge Graph...")
        
        # Add all entities as nodes
        print("📍 Adding entities as nodes...")
        all_entities = []
        for chunk_id, entities in chunk_entities.items():
            for entity in entities:
                self.add_entity(entity, chunk_id=chunk_id)
                all_entities.append(entity)
        
        print(f"   ✅ Added {self.entity_count} unique entities")
        
        # Add all relationships as edges
        print("🔗 Adding relationships as edges...")
        all_relationships = []
        for chunk_id, relationships in chunk_relationships.items():
            for rel in relationships:
                self.add_relationship(rel, chunk_id=chunk_id)
                all_relationships.append(rel)
        
        print(f"   ✅ Added {self.relationship_count} relationships")
        
        # Stats
        print(f"\n📊 Graph Statistics:")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {self.graph.number_of_edges()}")
        print(f"   Density: {nx.density(self.graph):.4f}")
        
        # Connected components
        if self.graph.number_of_nodes() > 0:
            weakly_connected = nx.number_weakly_connected_components(self.graph)
            print(f"   Connected components: {weakly_connected}")
    
    def get_neighbors(self, entity: str, direction: str = 'both') -> List[Tuple[str, str]]:
        """
        Get neighboring entities and their relations.
        
        Args:
            entity: Entity node ID
            direction: 'in', 'out', or 'both'
        
        Returns:
            List of (neighbor, relation) tuples
        """
        neighbors = []
        
        if entity not in self.graph:
            return neighbors
        
        if direction in ('out', 'both'):
            # Outgoing edges (entity -> neighbor)
            for target in self.graph.successors(entity):
                relation = self.graph[entity][target].get('relation', 'related_to')
                neighbors.append((target, relation))
        
        if direction in ('in', 'both'):
            # Incoming edges (neighbor -> entity)
            for source in self.graph.predecessors(entity):
                relation = self.graph[source][entity].get('relation', 'related_to')
                neighbors.append((source, relation))
        
        return neighbors
    
    def find_path(
        self,
        source: str,
        target: str,
        max_length: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
        
        Returns:
            List of paths (each path is list of nodes)
        """
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(
        self,
        entities: List[str],
        k_hop: int = 1
    ) -> nx.DiGraph:
        """
        Extract subgraph around given entities.
        
        Args:
            entities: List of entity node IDs
            k_hop: Number of hops to include
        
        Returns:
            Subgraph as NetworkX DiGraph
        """
        # Collect nodes within k hops
        nodes = set(entities)
        
        for _ in range(k_hop):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:
                    # Add neighbors
                    new_nodes.update(self.graph.successors(node))
                    new_nodes.update(self.graph.predecessors(node))
            nodes.update(new_nodes)
        
        # Extract subgraph
        return self.graph.subgraph(nodes).copy()
    
    def get_top_entities(self, n: int = 10, metric: str = 'degree') -> List[Tuple[str, float]]:
        """
        Get top N entities by centrality metric.
        
        Args:
            n: Number of top entities
            metric: 'degree', 'betweenness', or 'pagerank'
        
        Returns:
            List of (entity, score) tuples
        """
        if self.graph.number_of_nodes() == 0:
            return []
        
        if metric == 'degree':
            centrality = dict(self.graph.degree())
        elif metric == 'betweenness':
            centrality = nx.betweenness_centrality(self.graph)
        elif metric == 'pagerank':
            centrality = nx.pagerank(self.graph)
        else:
            centrality = dict(self.graph.degree())
        
        # Sort and return top N
        sorted_entities = sorted(
            centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_entities[:n]
    
    def save(self, filepath: str) -> None:
        """Save graph to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'entity_count': self.entity_count,
                'relationship_count': self.relationship_count
            }, f)
        
        print(f"💾 Graph saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load graph from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = data['graph']
        self.entity_count = data['entity_count']
        self.relationship_count = data['relationship_count']
        
        print(f"📂 Graph loaded from {filepath}")
    
    def __repr__(self):
        return (
            f"KnowledgeGraph("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
