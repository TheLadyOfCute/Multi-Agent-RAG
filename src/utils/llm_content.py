"""Helpers for normalizing LLM message content across providers."""

from __future__ import annotations

import json
from typing import Any


# 将不同格式的内容统一转换为纯文本
def content_to_text(content: Any) -> str:

    # 内容为空时返回空字符串
    if content is None:
        return ""

    # 如果内容本身就是字符串，直接返回
    if isinstance(content, str):
        return content

    # 如果内容是列表，逐项提取文本内容
    if isinstance(content, list):
        parts: list[str] = []

        # 遍历列表中的每一项
        for item in content:
            # 字符串项直接加入结果
            if isinstance(item, str):
                parts.append(item)

            # 字典项优先提取 text 或 content 字段
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    # 无可识别文本字段时，将字典序列化为 JSON 字符串
                    parts.append(json.dumps(item, ensure_ascii=False))

            # 其他类型统一转为字符串
            else:
                parts.append(str(item))

        # 过滤空内容，并用换行符拼接文本
        return "\n".join(part for part in parts if part)

    # 如果内容是字典，优先提取 text 或 content 字段
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]

        if isinstance(content.get("content"), str):
            return content["content"]

        # 无可识别文本字段时，将字典序列化为 JSON 字符串
        return json.dumps(content, ensure_ascii=False)

    # 其他类型统一转为字符串
    return str(content)


# 获取消息对象中的 content，并统一转换为纯文本
def message_content_to_text(message: Any) -> str:
    """Return normalized ``message.content`` text."""

    # 优先读取 message.content，不存在时直接使用 message 本身
    return content_to_text(getattr(message, "content", message))
