import logging

from ...core.common import ProviderRuntimeOptions, image_media_type, normalize_image_for_openai
from ...schemas import (
    AnthropicContentBlock,
    AnthropicImageBlock,
    AnthropicImageMediaType,
    AnthropicImageSource,
    AnthropicTextBlock,
    MessagePartImage,
    MessagePartText,
    MessageSnapshot,
)


def convert_content_blocks(
    message: MessageSnapshot,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
) -> list[AnthropicContentBlock]:
    blocks: list[AnthropicContentBlock] = []
    for part in message.parts:
        if isinstance(part, MessagePartText) and part.text:
            blocks.append(AnthropicTextBlock(text=part.text))
        elif message.role == "user" and isinstance(part, MessagePartImage):
            image_block = convert_image_block(part, options=options, logger=logger)
            if image_block is not None:
                blocks.append(image_block)
            elif options.invalid_image_policy == "placeholder":
                blocks.append(AnthropicTextBlock(text="[图片内容不可用]"))
    return blocks


def convert_image_block(
    part: MessagePartImage,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
) -> AnthropicImageBlock | None:
    normalized_image = normalize_image_for_openai(part, logger, options.image_limits)
    if normalized_image is None:
        if options.invalid_image_policy == "error":
            raise ValueError("图片数据无效，无法构建 Anthropic 图片消息片段")
        return None
    image_format, image_base64 = normalized_image
    media_type = anthropic_image_media_type(image_format)
    return AnthropicImageBlock(source=AnthropicImageSource(media_type=media_type, data=image_base64))


def anthropic_image_media_type(image_format: str | None) -> AnthropicImageMediaType:
    media_type = image_media_type(image_format)
    if media_type == "image/jpeg":
        return "image/jpeg"
    if media_type == "image/png":
        return "image/png"
    if media_type == "image/gif":
        return "image/gif"
    if media_type == "image/webp":
        return "image/webp"
    raise ValueError(f"Anthropic 不支持图片 media_type: {media_type}")
