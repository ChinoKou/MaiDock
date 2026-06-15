import logging
import re

from ...core.common import ProviderRuntimeOptions
from ...schemas import (
    ApiProviderSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..chat_completions_family.chat import ChatCompletionsMapper
from ..common.httpx import HttpxClientConfig, build_httpx_client_config, resolve_endpoint_path

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_API_PREFIX = "v1"
SILICONFLOW_PROVIDER_LABEL = "SiliconFlow"
SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT = "chat/completions"
QWEN_SERIES_PATTERN = re.compile(r"(^|/)qwen", re.IGNORECASE)


def build_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str,
    force_official_endpoint: bool,
) -> HttpxClientConfig:
    return build_httpx_client_config(
        api_provider,
        default_base_url=SILICONFLOW_BASE_URL,
        user_agent=user_agent,
        force_default_base_url=force_official_endpoint,
    )


def resolve_path(config: HttpxClientConfig, endpoint: str) -> str:
    return resolve_endpoint_path(config.base_url, api_prefix=SILICONFLOW_API_PREFIX, endpoint_path=endpoint)


def _create_mapper(*, options: ProviderRuntimeOptions, logger: logging.Logger) -> ChatCompletionsMapper:
    return ChatCompletionsMapper(
        options=options,
        logger=logger,
        provider_label=SILICONFLOW_PROVIDER_LABEL,
        raw_provider="siliconflow",
        policy_provider="siliconflow",
        tool_namespace="siliconflow",
        history_tool_prefix="siliconflow_history_tool",
        extract_tool_prefix="siliconflow_tool",
    )


def build_chat_body(
    request: ResponseRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
    stream: bool,
) -> tuple[dict, dict[str, str], dict]:
    mapper = _create_mapper(options=options, logger=logger)
    return mapper.build_request_body(request, stream=stream)


def convert_response(payload: dict, *, options: ProviderRuntimeOptions) -> ProviderResponse:
    mapper = _create_mapper(options=options, logger=logging.getLogger(__name__))
    return mapper.convert_response(payload)


def qwen_supports_dimensions(model: str) -> bool:
    return QWEN_SERIES_PATTERN.search(model) is not None
