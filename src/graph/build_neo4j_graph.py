"""Build the Neo4j graph in an isolated process.

Run with:
    python -m src.graph.build_neo4j_graph <chunks-json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_chunks(path: Path):
    from src.models.chunk import Chunk

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Chunk.from_dict(
            {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "doc_id": item.get("doc_id", "unknown"),
                "metadata": item.get("metadata") or {},
                "token_count": item.get("token_count", 0),
                "start_idx": item.get("start_idx", 0),
                "end_idx": item.get("end_idx", 0),
            }
        )
        for item in payload["chunks"]
    ]


def print_usage() -> None:
    print("Usage: python -m src.graph.build_neo4j_graph <chunks-json>")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print_usage()
        return 0
    if len(args) != 1:
        print_usage()
        return 2

    from src.config import get_settings
    from src.graph.entity_extractor import EntityExtractor
    from src.graph.neo4j_graph_store import Neo4jGraphStore
    from src.graph.relationship_extractor import RelationshipExtractor

    chunks = load_chunks(Path(args[0]))
    entity_extractor = EntityExtractor()
    rel_extractor = RelationshipExtractor()

    chunk_entities = {}
    chunk_relationships = {}
    relationship_method_counts = {
        "cooccurrence": 0,
        "pattern": 0,
        "dependency": 0,
    }

    for chunk in chunks:
        entities = entity_extractor.extract(chunk.text)
        chunk_entities[chunk.chunk_id] = entities

        relationships = (
            rel_extractor.extract_from_sentence(chunk.text, entities)
            if len(entities) >= 2
            else []
        )
        chunk_relationships[chunk.chunk_id] = relationships

        for relationship in relationships:
            method = getattr(relationship, "extraction_method", "unknown")
            if method in relationship_method_counts:
                relationship_method_counts[method] += 1

    print(
        "Graph relationship extraction stats: "
        f"cooccurrence={relationship_method_counts['cooccurrence']}, "
        f"pattern={relationship_method_counts['pattern']}, "
        f"dependency={relationship_method_counts['dependency']}"
    )

    settings = get_settings()
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    try:
        graph.build_from_chunks(chunks, chunk_entities, chunk_relationships)
        counts = graph.counts()
        print(f"Neo4j graph built: nodes={counts['nodes']}, edges={counts['edges']}")
    finally:
        graph.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
