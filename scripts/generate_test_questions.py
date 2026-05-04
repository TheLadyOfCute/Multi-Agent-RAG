"""Generate evaluation questions from persisted document content."""

from __future__ import annotations

import json
import random
import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings


# ──────────────────────────────────────────────────────────────────────
# Parameters: edit this dictionary to configure all defaults in one place.
# CLI arguments override these values.
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    # ── Input / Output ──
    "input_path": "data/Deep_learning.docx",       # source document (.txt, .md, .pdf, .docx)
    "output_path": "data/test_questions.json",     # generated questions JSON
    # ── ChromaDB ──
    "chroma_dir": "data/chroma_db",                # persistent ChromaDB directory
    "collection_name": "chunks",                   # collection name inside ChromaDB
    # ── LLM ──
    "model_id": "deepseek-v4-pro",                 # LLM model identifier
    "temperature": 0,                            # generation temperature (0.0–2.0)
    "max_tokens": 1000000,                            # max output tokens per LLM call
    # ── Question Generation ──
    "question_count": 5,                           # number of questions to generate
    "unanswerable_ratio": 0.4,                     # fraction of unanswerable questions (0.0–1.0)
    # ── Chunk Filtering ──
    "min_length": 100,                             # minimum character length for valid chunks
}
# ──────────────────────────────────────────────────────────────────────

VALID_QUESTION_TYPES = {
    "factual",
    "inferential",
    "definitional",
    "comparative",
    "causal",
}

ANSWERABLE_SYSTEM_PROMPT = """\
You are building a RAG evaluation dataset.
Given one full reference document, generate exactly one high-quality question-answer pair.

Rules:
1. The question must be answerable from the provided reference document only.
2. Ask the question as a normal user would ask it; never mention chunks, passages, context labels, or internal ids.
3. The ground_truth answer must be complete, accurate, and written in English.
4. The ground_truth answer must not mention chunks, passages, context labels, or internal ids.
5. question_type must be one of: factual, inferential, definitional, comparative, causal.
6. reference_contexts must be a non-empty JSON array of exact copied snippets from the reference document.
7. Do not paraphrase reference_contexts. Copy the minimum source text needed to support the answer.
8. Generate a mix of evidence scopes across repeated calls:
   - Some questions should be answerable from one localized part of the document.
   - Some questions should require combining evidence from two or more different parts of the document.
9. When the question requires multiple pieces of evidence, include each required source snippet as a separate item in reference_contexts.
10. For multi-evidence questions, prefer comparative, causal, or inferential questions that synthesize related ideas, contrasts, causes, consequences, trade-offs, or timelines from different parts of the document.
11. For single-evidence questions, prefer focused factual or definitional questions that can be fully supported by one concise source snippet.
12. Do not force every question to be multi-evidence. Choose naturally between single-evidence and multi-evidence questions based on the document content and the previous questions.

Return valid JSON only:
{"question": "...", "ground_truth": "...", "reference_contexts": ["..."], "question_type": "..."}
"""

UNANSWERABLE_SYSTEM_PROMPT = """\
You are building a RAG evaluation dataset.
Given one full reference document, generate exactly one plausible question related to deep learning,
but make sure the answer is NOT contained in the reference document.
Ask the question as a normal user would ask it; never mention chunks, passages,
context labels, or internal ids.
The question should be related to the topic, but it must require information absent from the document.

Return valid JSON only:
{"question": "...", "ground_truth": "unanswerable", "reference_contexts": [], "question_type": "unanswerable"}
"""

FORBIDDEN_CONTEXT_MARKER_RE = re.compile(
    r"\bchunk\s*\d+\b|\bchunk[_\s-]*id\b|\bgold[_\s-]*chunk\b|\bflat_[a-f0-9_]+\b",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_id = args.model_id
    question_count = args.question_count
    max_tokens = args.max_tokens
    input_path = args.input_path
    chroma_dir = args.chroma_dir
    collection_name = args.collection_name
    output_path = args.output_path
    min_length = args.min_length
    unanswerable_ratio = args.unanswerable_ratio
    temperature = args.temperature
    settings = get_settings()

    document_text = read_document(input_path)
    collection = open_chroma_collection(chroma_dir, collection_name)
    chunks = fetch_chunks(collection, min_length, source_filename=Path(input_path).name)
    if not chunks:
        raise RuntimeError("No valid chunks found.")

    llm = ChatOpenAI(
        model=model_id or settings.llm_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )

    results = []
    skipped = 0
    attempts = 0
    max_attempts = question_count * 10
    print(f"[INFO] document={input_path} chars={len(document_text)} chunks={len(chunks)}")

    while len(results) < question_count and attempts < max_attempts:
        attempts += 1
        qid = len(results) + 1
        make_unanswerable = random.random() < unanswerable_ratio
        mode = "unanswerable" if make_unanswerable else "answerable"
        print(f"[{qid:03d}] attempt={attempts} mode={mode} ", end="", flush=True)

        if make_unanswerable:
            qa = generate_unanswerable(llm, document_text, previous_questions(results))
            if not qa:
                skipped += 1
                print("SKIP")
                continue
            entry = build_unanswerable_entry(qid, qa)
        else:
            qa = generate_answerable(llm, document_text, previous_questions(results))
            if not qa:
                skipped += 1
                print("SKIP")
                continue
            entry = build_answerable_entry(qid, qa, chunks)
            if not entry:
                skipped += 1
                print("SKIP")
                continue

        if is_duplicate_question(entry["question"], previous_questions(results)):
            skipped += 1
            print("SKIP duplicate")
            continue

        results.append(entry)
        print(f"OK [{entry['question_type']}]")

    if len(results) < question_count:
        raise RuntimeError(
            f"Generated {len(results)} valid questions after {attempts} attempts; "
            f"skipped={skipped}."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] generated={len(results)} skipped={skipped} output={output.resolve()}")


def parse_args(argv: list[str] | None = None) -> Namespace:
    cfg = DEFAULT_CONFIG
    parser = ArgumentParser(description="Generate RAG evaluation questions from an indexed document.")
    parser.add_argument(
        "--input-path",
        default=cfg["input_path"],
        help="Source document path. Supports .txt, .md, .pdf, and .docx.",
    )
    parser.add_argument("--output-path", default=cfg["output_path"])
    parser.add_argument("--chroma-dir", default=cfg["chroma_dir"])
    parser.add_argument("--collection-name", default=cfg["collection_name"])
    parser.add_argument("--model-id", default=cfg["model_id"])
    parser.add_argument("--question-count", type=int, default=cfg["question_count"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--min-length", type=int, default=cfg["min_length"])
    parser.add_argument("--unanswerable-ratio", type=float, default=cfg["unanswerable_ratio"])
    parser.add_argument("--temperature", type=float, default=cfg["temperature"])
    return parser.parse_args(argv)


def read_document(path: str) -> str:
    from src.ingestion.document_loader import DocumentLoader

    document = DocumentLoader().load(path)
    text = document.text.strip()
    if not text:
        raise RuntimeError(f"Document is empty: {path}")
    return text


def open_chroma_collection(persist_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_collection(collection_name)


def fetch_chunks(
    collection: chromadb.Collection,
    min_length: int,
    source_filename: str | None = None,
) -> list[dict[str, Any]]:
    result = collection.get(include=["documents", "metadatas"])

    chunks = []
    for chunk_id, text, metadata in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        text = (text or "").strip()
        metadata = metadata or {}
        if source_filename and not chunk_matches_source(metadata, source_filename):
            continue
        if len(text) >= min_length:
            chunks.append({"chunk_id": chunk_id, "text": text, "metadata": metadata})
    return chunks


def chunk_matches_source(metadata: dict[str, Any], source_filename: str) -> bool:
    target = Path(source_filename).name
    for key in ("filename", "file_name", "document_name", "source_file", "source_filename"):
        value = metadata.get(key)
        if isinstance(value, str) and Path(value).name == target:
            return True
    for key in ("file_path", "path", "source_path", "source", "filepath"):
        value = metadata.get(key)
        if isinstance(value, str) and Path(value).name == target:
            return True
    return False


def generate_answerable(
    llm: ChatOpenAI, document_text: str, previous: list[str] | None = None
) -> dict[str, Any] | None:
    data = call_json_llm(llm, ANSWERABLE_SYSTEM_PROMPT, format_document(document_text, previous))
    if (
        not data
        or not data.get("question")
        or not data.get("ground_truth")
        or not data.get("reference_contexts")
    ):
        return None

    question_type = str(data.get("question_type", "")).lower().strip()
    if question_type not in VALID_QUESTION_TYPES:
        question_type = "factual"

    question = str(data["question"]).strip()
    ground_truth = str(data["ground_truth"]).strip()
    reference_contexts = [
        str(context).strip()
        for context in data.get("reference_contexts", [])
        if str(context).strip()
    ]
    if has_forbidden_context_marker(question) or has_forbidden_context_marker(ground_truth):
        return None
    if not reference_contexts:
        return None

    return {
        "question": question,
        "ground_truth": ground_truth,
        "reference_contexts": reference_contexts,
        "question_type": question_type,
    }

def generate_unanswerable(
    llm: ChatOpenAI, document_text: str, previous: list[str] | None = None
) -> dict[str, Any] | None:
    data = call_json_llm(llm, UNANSWERABLE_SYSTEM_PROMPT, format_document(document_text, previous))
    question = str((data or {}).get("question", "")).strip()
    if has_forbidden_context_marker(question):
        return None
    if not question:
        return None
    return {
        "question": question,
        "ground_truth": "unanswerable",
        "reference_contexts": [],
        "question_type": "unanswerable",
    }


def call_json_llm(llm: ChatOpenAI, system_prompt: str, user_text: str) -> dict[str, Any] | None:
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Reference document:\n{user_text}"),
            ]
        )
        content = str(response.content).strip()
        if content.startswith("```"):
            content = "\n".join(line for line in content.splitlines() if not line.startswith("```"))
        return json.loads(content)
    except Exception as exc:
        print(f"LLM/JSON error: {exc}")
        return None


def format_document(document_text: str, previous: list[str] | None = None) -> str:
    text = document_text.strip()
    previous = [question.strip() for question in (previous or []) if question.strip()]
    if not previous:
        return text

    numbered = "\n".join(f"{index}. {question}" for index, question in enumerate(previous, 1))
    return (
        f"{text}\n\n"
        "Previous questions:\n"
        f"{numbered}\n\n"
        "Do not generate a question that is the same as, semantically equivalent to, "
        "or only a wording variation of any previous question."
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_question(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def is_duplicate_question(question: str, previous: list[str]) -> bool:
    normalized = normalize_question(question)
    if not normalized:
        return True
    return normalized in {normalize_question(item) for item in previous}


def previous_questions(results: list[dict[str, Any]]) -> list[str]:
    return [str(entry.get("question", "")).strip() for entry in results if entry.get("question")]


def match_reference_contexts_to_chunk_ids(
    reference_contexts: list[str], chunks: list[dict[str, Any]]
) -> list[str]:
    normalized_contexts = [
        normalize_whitespace(context)
        for context in reference_contexts
        if normalize_whitespace(context)
    ]
    if not normalized_contexts:
        return []

    matched_ids = []
    seen = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        chunk_text = normalize_whitespace(str(chunk.get("text") or ""))
        if any(context in chunk_text for context in normalized_contexts):
            seen.add(chunk_id)
            matched_ids.append(chunk_id)
    return matched_ids


def has_forbidden_context_marker(text: str) -> bool:
    return bool(FORBIDDEN_CONTEXT_MARKER_RE.search(text))


def build_answerable_entry(
    qid: int, qa: dict[str, Any], chunks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    reference_contexts = [
        str(context).strip()
        for context in qa.get("reference_contexts", [])
        if str(context).strip()
    ]
    gold_chunk_ids = match_reference_contexts_to_chunk_ids(reference_contexts, chunks)
    if not reference_contexts or not gold_chunk_ids:
        return None

    return {
        "id": f"q{qid:03d}",
        "question": qa["question"],
        "ground_truth": qa["ground_truth"],
        "reference_contexts": reference_contexts,
        "gold_chunk_ids": gold_chunk_ids,
        "question_type": qa["question_type"],
    }


def build_unanswerable_entry(qid: int, qa: dict[str, Any] | str) -> dict[str, Any]:
    question = qa.get("question", "") if isinstance(qa, dict) else qa
    return {
        "id": f"q{qid:03d}",
        "question": str(question).strip(),
        "ground_truth": "unanswerable",
        "reference_contexts": [],
        "gold_chunk_ids": [],
        "question_type": "unanswerable",
    }


if __name__ == "__main__":
    main()
