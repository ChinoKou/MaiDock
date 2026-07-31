import logging

from ...core.common import RuntimeOptionsView
from ...schemas import MessagePartImage, MessageSnapshot
from ..chat_completions_family import multimodal as family_multimodal
from ...core.json_types import JsonValue


def message_content_text(value: object) -> str | None:
    """从 Chat Completions message.content 中提取文本。"""
    return family_multimodal.message_content_text(value)


def build_image_content(
    part: MessagePartImage,
    *,
    options: RuntimeOptionsView,
    logger: logging.Logger,
) -> dict[str, JsonValue] | None:
    """构造 SiliconFlow image_url content part。"""
    return family_multimodal.build_image_content(part, logger=logger, options=options)


def convert_message_content(
    message: MessageSnapshot,
    *,
    options: RuntimeOptionsView,
    logger: logging.Logger,
) -> str | list[dict[str, JsonValue]] | None:
    """把 Host 消息内容转给 Chat Completions family 标准实现。"""
    return family_multimodal.convert_message_content(message, options=options, logger=logger)
