"""Shared paths used by infrastructure and use cases."""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings


_settings = get_settings()

BM25_INDEX_PATH = Path(_settings.bm25_index_path)
CHROMA_DIR = _settings.chroma_persist_dir
UPLOAD_DIR = Path(_settings.upload_dir)
