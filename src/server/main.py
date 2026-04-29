"""Primary FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.server.routes.chat import router as chat_router
from src.server.routes.data import router as data_router
from src.server.routes.documents import router as documents_router
from src.server.routes.evaluation import router as evaluation_router
from src.server.routes.system import router as system_router
from src.server.utils.lifespan import app_lifespan

# 鎷艰 FastAPI
# 鐪熸鐨勪笟鍔￠€昏緫宸茬粡涓嬫矇鍒?route -> use case -> infrastructure銆?
app = FastAPI(title="Multi-Agent RAG API", version="1.0.0", lifespan=app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 璺敱娉ㄥ唽椤哄簭娌℃湁鐗规畩璇箟锛屾墍鏈夋ā鍧楃粺涓€鎸傚埌鍚屼竴涓?app 涓娿€?
app.include_router(system_router)
app.include_router(documents_router)
app.include_router(data_router)
app.include_router(chat_router)
app.include_router(evaluation_router)


def main() -> None:
    import logging

    import uvicorn

    from src.config import get_settings

    settings = get_settings()
    # Uvicorn configures handlers for uvicorn.* loggers by default; use that so messages show up.
    logger = logging.getLogger("uvicorn.error")
    logger.info("Starting API (bind=%s:%s reload=%s)", settings.api_host, settings.api_port, settings.api_reload)
    uvicorn.run(
        "src.server.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        reload_excludes=["data/*", ".git/*", "*.pkl", "__pycache__"],
        access_log=False,
    )


if __name__ == "__main__":
    main()

