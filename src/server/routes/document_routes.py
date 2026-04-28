"""Document management endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.server.utils.dependencies import (
    get_delete_document_use_case,
    get_list_documents_use_case,
    get_process_uploaded_document_use_case,
    get_preview_document_use_case,
    get_save_upload_use_case,
    get_task_registry,
)

# 创建文档管理路由，并统一设置接口前缀和标签
router = APIRouter(prefix="/api/documents", tags=["documents"])


# 获取当前已加载的文档列表
@router.get("")
def documents() -> dict[str, Any]:
    # 执行文档列表查询用例，并包装为接口返回格式
    return {"documents": get_list_documents_use_case().execute()}


# 上传文档并提交后台处理任务
@router.post("")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:
    # 提取上传文件后缀，并统一转为小写
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")

    # 校验文件类型，只允许处理支持的文档格式
    if suffix not in {"pdf", "docx", "txt"}:
        raise HTTPException(status_code=400, detail="Only pdf, docx, and txt files are supported.")

    # 先将上传文件保存到本地临时或持久目录
    saved_path = get_save_upload_use_case().execute(file.filename or "upload.txt", file.file)

    # 定义后台任务
    def runner(task_id, update_task):
        # 执行较重的文档处理流程，并通过 update_task 回写进度
        return get_process_uploaded_document_use_case().execute(saved_path, update_task, task_id)

    # 提交文档上传处理任务，并返回任务 ID 供前端轮询
    return {"task_id": get_task_registry().submit_task("document_upload", runner)}


# 删除指定名称的文档
@router.delete("/{name}")
def delete_document(name: str) -> dict[str, Any]:
    # 执行文档删除用例，并返回删除结果
    return get_delete_document_use_case().execute(name)


# 获取指定文档的预览内容
@router.get("/{name}/preview")
def preview_document(name: str) -> dict[str, Any]:
    try:
        # 执行文档预览用例
        return get_preview_document_use_case().execute(name)
    except FileNotFoundError as exc:
        # 文档不存在时转换为 404 HTTP 错误
        raise HTTPException(status_code=404, detail=str(exc)) from exc