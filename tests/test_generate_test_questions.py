import importlib.util
import sys
import types
from pathlib import Path


def load_script():
    replacements = {
        "chromadb": types.SimpleNamespace(Collection=object),
        "chromadb.config": types.SimpleNamespace(Settings=lambda **_: object()),
        "langchain_core.messages": types.SimpleNamespace(
            HumanMessage=object,
            SystemMessage=object,
        ),
        "langchain_openai": types.SimpleNamespace(ChatOpenAI=object),
        "src.config": types.SimpleNamespace(
            get_settings=lambda: types.SimpleNamespace(),
        ),
    }
    originals = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_test_questions.py"
    spec = importlib.util.spec_from_file_location("generate_test_questions", path)
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


def test_format_group_does_not_expose_chunk_ids_or_labels():
    script = load_script()
    text = script.format_group(
        [
            {"chunk_id": "flat_abc_0", "text": "First fact."},
            {"chunk_id": "flat_def_4", "text": "Second fact."},
        ]
    )

    assert "First fact." in text
    assert "Second fact." in text
    assert "flat_" not in text
    assert "chunk" not in text.lower()


def test_format_document_does_not_expose_chunk_ids_or_labels():
    script = load_script()

    text = script.format_document("Deep learning uses hierarchical representations.")

    assert text == "Deep learning uses hierarchical representations."
    assert "flat_" not in text
    assert "chunk" not in text.lower()


def test_reference_context_matching_returns_expected_chunk_id():
    script = load_script()
    chunks = [
        {"chunk_id": "flat_abc_0", "text": "First fact about neural networks."},
        {"chunk_id": "flat_def_1", "text": "Second fact about transformers."},
    ]

    matches = script.match_reference_contexts_to_chunk_ids(
        ["fact about transformers"],
        chunks,
    )

    assert matches == ["flat_def_1"]


def test_reference_context_matching_dedupes_in_chunk_order():
    script = load_script()
    chunks = [
        {"chunk_id": "flat_abc_0", "text": "Alpha shared fact."},
        {"chunk_id": "flat_def_1", "text": "Beta shared fact."},
    ]

    matches = script.match_reference_contexts_to_chunk_ids(
        ["shared fact", "Alpha shared fact."],
        chunks,
    )

    assert matches == ["flat_abc_0", "flat_def_1"]


def test_build_answerable_entry_rejects_unmatched_contexts():
    script = load_script()
    qa = {
        "question": "What do transformers use?",
        "ground_truth": "Transformers use attention.",
        "reference_contexts": ["This exact text is not indexed."],
        "question_type": "factual",
    }

    entry = script.build_answerable_entry(1, qa, [])

    assert entry is None


def test_duplicate_questions_are_detected_after_normalization():
    script = load_script()

    assert script.is_duplicate_question(
        "What is retrieval augmented generation?",
        ["What is retrieval-augmented generation?"],
    )
    assert not script.is_duplicate_question(
        "Why can dense retrieval fail on numerical constraints?",
        ["What is retrieval-augmented generation?"],
    )


def test_previous_questions_are_added_to_prompt():
    script = load_script()

    prompt = script.format_document(
        "Deep learning text.",
        ["What is retrieval-augmented generation?"],
    )

    assert "Deep learning text." in prompt
    assert "Previous questions:" in prompt
    assert "What is retrieval-augmented generation?" in prompt


def test_build_unanswerable_entry_ignores_llm_reference_contexts():
    script = load_script()
    entry = script.build_unanswerable_entry(
        1,
        {
            "question": "What is not in the text?",
            "ground_truth": "Some hallucinated answer.",
            "reference_contexts": ["Not allowed."],
            "question_type": "factual",
        },
    )

    assert entry["ground_truth"] == "unanswerable"
    assert entry["reference_contexts"] == []
    assert entry["gold_chunk_ids"] == []
    assert entry["question_type"] == "unanswerable"


def test_build_answerable_entry_uses_ground_truth_field():
    script = load_script()
    chunks = [
        {"chunk_id": "flat_def_1", "text": "Transformers use attention mechanisms."},
    ]
    qa = {
        "question": "What do transformers use?",
        "ground_truth": "Transformers use attention mechanisms.",
        "reference_contexts": ["attention mechanisms"],
        "question_type": "factual",
    }

    entry = script.build_answerable_entry(1, qa, chunks)

    assert entry["ground_truth"] == "Transformers use attention mechanisms."
    assert "reference" not in entry


def test_forbidden_context_markers_are_detected():
    script = load_script()

    assert script.has_forbidden_context_marker("The concept from chunk 1 is representation learning.")
    assert script.has_forbidden_context_marker("Use flat_c06dbe64e5924f6f391bcf833e3b934e_4.")
    assert not script.has_forbidden_context_marker("Why does fine-tuning reduce deployment cost?")


def test_parse_args_accepts_pdf_docx_input_path():
    script = load_script()

    pdf_args = script.parse_args(["--input-path", "data/example.pdf"])
    docx_args = script.parse_args(["--input-path", "data/example.docx"])

    assert pdf_args.input_path == "data/example.pdf"
    assert docx_args.input_path == "data/example.docx"


def test_read_document_uses_document_loader_for_pdf_and_docx(monkeypatch):
    script = load_script()
    loaded_paths = []

    class FakeDocument:
        text = " Extracted document text. "

    class FakeDocumentLoader:
        def load(self, path):
            loaded_paths.append(path)
            return FakeDocument()

    fake_loader_module = types.SimpleNamespace(DocumentLoader=FakeDocumentLoader)
    monkeypatch.setitem(sys.modules, "src.ingestion.document_loader", fake_loader_module)

    assert script.read_document("upload.pdf") == "Extracted document text."
    assert script.read_document("upload.docx") == "Extracted document text."
    assert loaded_paths == ["upload.pdf", "upload.docx"]


def test_fetch_chunks_filters_to_uploaded_filename():
    script = load_script()

    class FakeCollection:
        def get(self, include):
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["c1", "c2", "c3"],
                "documents": [
                    "A" * 120,
                    "B" * 120,
                    "too short",
                ],
                "metadatas": [
                    {"filename": "paper.pdf"},
                    {"file_path": "data/uploads/other.docx"},
                    {"filename": "paper.pdf"},
                ],
            }

    chunks = script.fetch_chunks(FakeCollection(), min_length=100, source_filename="paper.pdf")

    assert chunks == [
        {"chunk_id": "c1", "text": "A" * 120, "metadata": {"filename": "paper.pdf"}},
    ]
