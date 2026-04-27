from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_health_and_initial_state(monkeypatch) -> None:
    from src.app import main
    from src.app.dependencies import get_restore_document_state_use_case, get_runtime_state

    state = get_runtime_state()
    state.reset_runtime()
    monkeypatch.setattr(get_restore_document_state_use_case(), "restore_persisted_state", lambda: None)

    with TestClient(main.app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        response = client.get("/api/state")

    assert response.status_code == 200
    body = response.json()
    assert body["rag_initialized"] is False
    assert body["document_count"] == 0
    assert body["message_count"] == 0


def test_chat_guard_requires_uploaded_documents(monkeypatch) -> None:
    from src.app import main
    from src.app.dependencies import get_restore_document_state_use_case, get_runtime_state

    state = get_runtime_state()
    state.reset_runtime()
    monkeypatch.setattr(get_restore_document_state_use_case(), "restore_persisted_state", lambda: None)

    with TestClient(main.app) as client:
        response = client.post("/api/chat/messages", json={"query": "hello"})

    assert response.status_code == 409
    assert "请先上传文档" in response.json()["detail"]


def test_document_upload_creates_task(monkeypatch, tmp_path) -> None:
    from src.app import main
    from src.app.dependencies import get_restore_document_state_use_case, get_save_upload_use_case, get_task_registry
    from src.app.dependencies import get_runtime_state

    state = get_runtime_state()
    state.reset_runtime()
    monkeypatch.setattr(get_restore_document_state_use_case(), "restore_persisted_state", lambda: None)
    monkeypatch.setattr(get_save_upload_use_case(), "execute", lambda filename, file_obj: tmp_path / filename)

    def fake_submit_task(task_type, runner):
        assert task_type == "document_upload"
        return "task-123"

    monkeypatch.setattr(get_task_registry(), "submit_task", fake_submit_task)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("sample.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-123"}


def test_chat_task_endpoint_returns_task_id(monkeypatch) -> None:
    from src.app import main
    from src.app.dependencies import get_restore_document_state_use_case, get_run_chat_query_use_case, get_runtime_state

    state = get_runtime_state()
    state.reset_runtime()
    monkeypatch.setattr(get_restore_document_state_use_case(), "restore_persisted_state", lambda: None)
    monkeypatch.setattr(get_run_chat_query_use_case(), "submit_query_task", lambda query: "chat-task-1")

    with TestClient(main.app) as client:
        response = client.post("/api/chat/tasks", json={"query": "hello"})

    assert response.status_code == 200
    assert response.json() == {"task_id": "chat-task-1"}


def test_evaluation_questions_normalize_list(tmp_path) -> None:
    from src.use_cases.evaluation import LoadTestQuestionsUseCase

    test_file = tmp_path / "questions.json"
    test_file.write_text(
        '[{"id": "q1", "question": "What is RAG?", "ground_truth": "retrieval"}]',
        encoding="utf-8",
    )

    questions = LoadTestQuestionsUseCase().execute(test_file)

    assert questions == [
        {
            "id": "q1",
            "question": "What is RAG?",
            "question_type": "",
            "reference": "retrieval",
        }
    ]
