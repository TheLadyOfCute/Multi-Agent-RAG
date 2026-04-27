"""Storage factory helpers."""

from __future__ import annotations

from typing import Any

from src.app.paths import CHROMA_DIR


def open_vector_store():
    # 延迟导入 Chroma 向量库，避免轻量 API 导入时强依赖 Chroma
    from src.storage.chroma_store import ChromaVectorStore

    # 使用配置的持久化目录创建向量库实例
    return ChromaVectorStore(persist_directory=CHROMA_DIR)


def close_vector_store(vector_store: Any) -> None:
    # 判断向量库对象是否存在且支持 close 方法
    if vector_store is not None and hasattr(vector_store, "close"):
        try:
            # 关闭向量库连接并释放资源
            vector_store.close()
        except Exception as exc:
            # 关闭失败时输出错误信息，避免影响主流程
            print(f"ChromaDB close failed: {exc}", flush=True)