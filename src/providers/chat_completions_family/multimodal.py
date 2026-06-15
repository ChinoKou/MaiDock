import logging
from collections.abc import Callable

from ...core.common import ProviderRuntimeOptions, message_text
from ...core.json_types import json_list_or_none, json_mapping_or_none
from ...schemas import MessagePartImage, MessagePartText, MessageSnapshot
from ..common.multimodal import image_data_url_or_none


def message_content_text(value: object) -> str | None:
    """从 Chat Completions message.content 中提取文本。"""
    if isinstance(value, str):
        return value
    content_parts = json_list_or_none(value)
    if content_parts is None:
        return None
    parts: list[str] = []
    for item in content_parts:
        item_mapping = json_mapping_or_none(item)
        if item_mapping is None:
            continue
        text = item_mapping.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts) or None


def build_image_content(
    part: MessagePartImage,
    *,
    logger: logging.Logger,
    options: ProviderRuntimeOptions,
) -> dict | None:
    """构造标准 image_url content part。"""
    data_url = image_data_url_or_none(part, logger=logger, options=options)
    if data_url:
        return {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}}
    return None


def convert_message_content(
    message: MessageSnapshot,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
    image_builder: Callable[[MessagePartImage], dict | None] | None = None,
) -> str | list[dict] | None:
    """把 Host 消息内容转换为标准 Chat Completions content。"""
    if message.role != "user":
        content = message_text(message)
        return content if content else None

    has_image_part = any(isinstance(part, MessagePartImage) for part in message.parts)
    if not has_image_part:
        content = message_text(message)
        return content if content else None

    build_image = image_builder or (lambda part: build_image_content(part, logger=logger, options=options))
    converted_parts: list[dict] = []
    for part in message.parts:
        if isinstance(part, MessagePartText) and part.text:
            converted_parts.append({"type": "text", "text": part.text})
        elif isinstance(part, MessagePartImage):
            image_block = build_image(part)
            if image_block is not None:
                converted_parts.append(image_block)
            elif options.invalid_image_policy == "placeholder":
                converted_parts.append({"type": "text", "text": "[图片内容不可用]"})
    return converted_parts or None
