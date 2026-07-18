import logging

from ...core.common import ProviderRuntimeOptions
from ...schemas.host_snapshots import MessagePartImage, MessagePartText, MessageSnapshot
from ...schemas.responses_compat import (
    OpenAIInputImageBlock,
    OpenAIInputTextBlock,
    OpenAIResponseOutputItem,
    OpenAIResponseSnapshot,
)
from ..common.multimodal import image_data_url_or_none


def convert_user_content_parts(
    message: MessageSnapshot,
    *,
    logger: logging.Logger,
    options: ProviderRuntimeOptions,
) -> list[OpenAIInputTextBlock | OpenAIInputImageBlock]:
    parts: list[OpenAIInputTextBlock | OpenAIInputImageBlock] = []
    for part in message.parts:
        if isinstance(part, MessagePartText) and part.text:
            parts.append(OpenAIInputTextBlock(text=part.text))
        elif message.role == "user" and isinstance(part, MessagePartImage):
            data_url = image_data_url_or_none(part, logger=logger, options=options)
            if data_url:
                parts.append(OpenAIInputImageBlock(image_url=data_url, detail="auto"))
            elif options.invalid_image_policy == "placeholder":
                parts.append(OpenAIInputTextBlock(text="[图片内容不可用]"))
    return parts


def extract_text_content(response_model: OpenAIResponseSnapshot) -> str:
    if response_model.output_text:
        return response_model.output_text
    chunks: list[str] = []
    for item in response_model.output:
        if item.type != "message":
            continue
        for block in item.content:
            if block.type in {"output_text", "text"} and block.text:
                chunks.append(block.text)
    return "".join(chunks)


def extract_reasoning_content(output: list[OpenAIResponseOutputItem]) -> str | None:
    chunks: list[str] = []
    for item in output:
        if item.type not in {"reasoning", "reasoning_summary"}:
            continue
        for block in item.summary:
            if block.text:
                chunks.append(block.text)
        for block in item.content:
            if block.text:
                chunks.append(block.text)
    return "".join(chunks) or None
