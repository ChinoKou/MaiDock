import logging

from ...core.common import ProviderRuntimeOptions
from ...schemas import MessagePartImage, MessageSnapshot
from ..chat_completions_family import multimodal as family_multimodal


def message_content_text(value: object) -> str | None:
    """从 Chat Completions message.content 中提取文本。"""
    return family_multimodal.message_content_text(value)


def build_image_content(
    part: MessagePartImage,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
) -> dict | None:
    """构造 Mimo image_url content part。"""
    return family_multimodal.build_image_content(part, logger=logger, options=options)


def convert_message_content(
    message: MessageSnapshot,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
) -> str | list[dict] | None:
    """把 Host 消息内容转给 Chat Completions family 标准实现。"""
    return family_multimodal.convert_message_content(message, options=options, logger=logger)
