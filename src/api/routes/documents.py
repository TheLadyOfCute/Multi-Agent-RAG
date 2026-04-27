"""Document management endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.app.dependencies import (
    get_delete_document_use_case,
    get_list_documents_use_case,
    get_process_uploaded_document_use_case,
    get_preview_document_use_case,
    get_save_upload_use_case,
    get_task_registry,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def documents() -> dict[str, Any]:
    return {"documents": get_list_documents_use_case().execute()}


@router.post("")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in {"pdf", "docx", "txt"}:
        raise HTTPException(status_code=400, detail="Only pdf, docx, and txt files are supported.")
    saved_path = get_save_upload_use_case().execute(file.filename or "upload.txt", file.file)

    def runner(task_id, update_task):
        # 真正的导入流程比较重，所以放到后台任务里处理。
        return get_process_uploaded_document_use_case().execute(saved_path, update_task, task_id)

    return {"task_id": get_task_registry().submit_task("document_upload", runner)}


@router.delete("/{name}")
def delete_document(name: str) -> dict[str, Any]:
    return get_delete_document_use_case().execute(name)


@router.get("/{name}/preview")
def preview_document(name: str) -> dict[str, Any]:
    try:
        return get_preview_document_use_case().execute(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
