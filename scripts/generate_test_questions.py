"""Generate evaluation questions from persisted document content."""

from __future__ import annotations

import json
import random
import re
import sys
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


def main() -> None:
    model_id = "deepseek-v4-pro"
    question_count = 5
    max_tokens = 8192
    input_path = "data/deep_learning.txt"
    chroma_dir = "data/chroma_db"
    collection_name = "chunks"
    output_path = "data/test_questions.json"
    min_length = 100
    group_size = 3
    min_gold_chunks = 1
    unanswerable_ratio = 0.4
    temperature = 0.7
    _ = (group_size, min_gold_chunks)
    settings = get_settings()

    document_text = read_document(input_path)
    collection = open_chroma_collection(chroma_dir, collection_name)
    chunks = fetch_chunks(collection, min_length)
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


def read_document(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Document is empty: {path}")
    return text


def open_chroma_collection(persist_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_collection(collection_name)


def fetch_chunks(collection: chromadb.Collection, min_length: int) -> list[dict[str, Any]]:
    result = collection.get(include=["documents", "metadatas"])

    chunks = []
    for chunk_id, text, metadata in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        text = (text or "").strip()
        metadata = metadata or {}
        if len(text) >= min_length:
            chunks.append({"chunk_id": chunk_id, "text": text, "metadata": metadata})
    return chunks


def build_context_groups(chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in dedupe_chunks_by_id(chunks):
        metadata = chunk.get("metadata") or {}
        source = str(
            metadata.get("filename")
            or metadata.get("source")
            or metadata.get("doc_id")
            or "unknown"
        )
        by_source.setdefault(source, []).append(chunk)

    groups = []
    for source_chunks in by_source.values():
        source_chunks.sort(key=chunk_sort_key)
        if source_chunks:
            groups.append(source_chunks)
    return groups


def dedupe_chunks_by_id(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(chunk)
    return unique


def sample_context_groups(
    groups: list[list[dict[str, Any]]],
    question_count: int,
    min_gold_chunks: int,
    max_gold_chunks: int,
) -> list[list[dict[str, Any]]]:
    eligible = [group for group in groups if len(group) >= min_gold_chunks]
    if not eligible:
        raise RuntimeError(f"No context groups have at least {min_gold_chunks} chunks.")

    sample = []
    for _ in range(question_count):
        source_chunks = random.choice(eligible)
        gold_count = random.randint(min_gold_chunks, min(max_gold_chunks, len(source_chunks)))
        selected = random.sample(source_chunks, gold_count)
        selected.sort(key=chunk_sort_key)
        sample.append(selected)
    return sample


def chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, int, str]:
    metadata = chunk.get("metadata") or {}
    return (
        int(metadata.get("start_idx") or metadata.get("start_char") or 0),
        int(metadata.get("end_idx") or metadata.get("end_char") or 0),
        str(chunk.get("chunk_id") or ""),
    )


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


def format_group(group: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(chunk["text"] for chunk in group)


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
