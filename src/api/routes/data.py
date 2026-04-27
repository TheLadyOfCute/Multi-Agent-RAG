"""Data maintenance endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.app.dependencies import get_clear_all_data_use_case

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/clear")
def clear_data() -> dict[str, Any]:
    return get_clear_all_data_use_case().execute()
