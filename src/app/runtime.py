"""Runtime environment safeguards shared by API entrypoints."""

from __future__ import annotations

import logging
import os
import sys


def configure_runtime() -> None:
    """
    在导入大型第三方库之前，配置运行时环境。

    主要作用：
    1. 解决 Windows 控制台输出中文或特殊字符时可能出现的编码问题。
    2. 降低 transformers、tokenizers、uvicorn 等库的日志噪音。
    3. 避免 tokenizer 并行警告影响终端输出。
    """

    # 统一配置标准输出 stdout 和标准错误 stderr 的编码
    # 在 Windows 下，默认控制台编码可能不是 UTF-8，容易导致中文乱码
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # 降低 Hugging Face Transformers 的日志级别
    # 只显示 error，避免大量 warning/info 干扰终端输出
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    # 禁用 tokenizers 的并行提示
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 进一步从 logging 层面压低 transformers 日志级别
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # 降低 uvicorn 访问日志级别 只显示 warning 及以上日志，避免每个 HTTP 请求都刷屏 
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
