"""Evaluation endpoints for question files and RAGAS tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import RagasRequest
from src.app.dependencies import (
    get_load_test_questions_use_case,
    get_save_test_questions_upload_use_case,
    get_submit_ragas_evaluation_use_case,
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/questions")
def evaluation_questions(test_file: str = "data/test_questions.json") -> dict[str, Any]:
    path = Path(test_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Test questions file not found: {path}")
    try:
        return {"test_file": str(path), "questions": get_load_test_questions_use_case().execute(path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/questions")
async def upload_evaluation_questions(file: UploadFile = File(...)) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    content = await file.read()
    try:
        return get_save_test_questions_upload_use_case().execute(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ragas")
def start_ragas(request: RagasRequest) -> dict[str, str]:
    if not Path(request.test_file).exists():
        raise HTTPException(status_code=404, detail=f"Test file not found: {request.test_file}")
    return {
        "task_id": get_submit_ragas_evaluation_use_case().submit(
            test_file=request.test_file,
            reuse_rag_outputs=request.reuse_rag_outputs,
        )
    }
