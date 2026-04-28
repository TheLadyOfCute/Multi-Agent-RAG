"""Chat conversation endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from src.server.schemas import ChatRequest
from src.server.utils.dependencies import (
    get_clear_messages_use_case,
    get_export_chat_history_use_case,
    get_get_messages_use_case,
    get_run_chat_query_use_case,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.get("/messages")
def chat_messages() -> dict[str, Any]:
    return {"messages": get_get_messages_use_case().execute()}


#测试用
@router.post("/messages")
def create_chat_message(request: ChatRequest) -> dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        # 同步接口：当前请求会一直等待到完整答案生成完成。
        return get_run_chat_query_use_case().process_query(query)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/tasks")
def create_chat_task(request: ChatRequest) -> dict[str, str]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        # 异步接口：先返回 task_id，再由前端轮询任务状态。
        return {"task_id": get_run_chat_query_use_case().submit_query_task(query)}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/messages")
def delete_chat_messages() -> dict[str, bool]:
    return get_clear_messages_use_case().execute()


@router.get("/export", response_class=PlainTextResponse)
def export_chat() -> str:
    return get_export_chat_history_use_case().execute()
