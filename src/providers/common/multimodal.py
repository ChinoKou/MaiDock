import logging

from ...core.common import ProviderRuntimeOptions, image_data_url
from ...schemas import MessagePartImage


def image_data_url_or_none(
    part: MessagePartImage,
    *,
    logger: logging.Logger,
    options: ProviderRuntimeOptions,
) -> str | None:
    return image_data_url(part, logger, options.invalid_image_policy, options.image_limits)
