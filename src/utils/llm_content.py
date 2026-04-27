"""Helpers for normalizing LLM message content across providers."""

from __future__ import annotations

import json
from typing import Any


def content_to_text(content: Any) -> str:
    """Convert OpenAI/LangChain content variants into plain text.

    Some OpenAI-compatible providers return assistant content as a list of
    blocks, for example ``[{"text": "..."}]``. Most agents in this project
    expect plain text, so normalize once at the boundary.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def message_content_to_text(message: Any) -> str:
    """Return normalized ``message.content`` text."""
    return content_to_text(getattr(message, "content", message))
