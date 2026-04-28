import sys
import types
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_graph_retrieval_module():
    replacements = {
        "langchain_openai": types.SimpleNamespace(ChatOpenAI=object),
        "spacy": types.SimpleNamespace(load=lambda *args, **kwargs: None),
    }
    originals = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    path = Path(__file__).resolve().parents[1] / "src" / "retrieval" / "graph_retrieval.py"
    spec = importlib.util.spec_from_file_location("graph_retrieval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


class DummyEntity:
    def __init__(self, normalized):
        self.normalized = normalized


class DummyGraph:
    def match_entities(self, entities):
        return [entity for entity in entities if entity == "rag"]


def test_query_entity_extraction_matches_graph_using_entity_extractor(monkeypatch):
    module = load_graph_retrieval_module()
    retrieval = module.GraphRetrieval(knowledge_graph=DummyGraph(), vector_store=None)

    class FakeExtractor:
        def extract(self, query):
            return [DummyEntity("rag"), DummyEntity("rag"), DummyEntity("python")]

    monkeypatch.setattr(module, "EntityExtractor", FakeExtractor, raising=False)

    raw_entities = retrieval._extract_query_entities("How does RAG work in Python?")
    entities = retrieval._filter_entities_in_graph(raw_entities)

    assert raw_entities == ["rag", "python"]
    assert entities == ["rag"]
