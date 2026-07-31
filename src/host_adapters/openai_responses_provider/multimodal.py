import logging

from ...core.common import RuntimeOptionsView
from ...schemas.host_snapshots import MessageSnapshot
from ...schemas.responses_compat import (
    OpenAIInputImageBlock,
    OpenAIInputTextBlock,
    OpenAIResponseOutputItem,
    OpenAIResponseSnapshot,
)
from ..responses_family.multimodal import convert_user_content_parts as _family_convert_user_content_parts
from ..responses_family.multimodal import extract_reasoning_content as _family_extract_reasoning_content
from ..responses_family.multimodal import extract_text_content as _family_extract_text_content


def convert_user_content_parts(
    message: MessageSnapshot,
    *,
    logger: logging.Logger,
    options: RuntimeOptionsView,
) -> list[OpenAIInputTextBlock | OpenAIInputImageBlock]:
    return _family_convert_user_content_parts(message, logger=logger, options=options)


def extract_text_content(response_model: OpenAIResponseSnapshot) -> str:
    return _family_extract_text_content(response_model)


def extract_reasoning_content(output: list[OpenAIResponseOutputItem]) -> str | None:
    return _family_extract_reasoning_content(output)
