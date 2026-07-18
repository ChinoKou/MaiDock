import logging
import uuid
from collections.abc import Mapping

import httpx

from ...core.common import ProviderRuntimeOptions
from ...core.json_types import json_list_or_none, json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...schemas import (
    ApiProviderSnapshot,
    MessageSnapshot,
    OpenAIInputImageBlock,
    OpenAIInputTextBlock,
    OpenAIResponseOutputItem,
    OpenAIResponseSnapshot,
    OpenAIResponsesTool,
    ProviderToolCall,
    ToolOptionSnapshot,
)
from ..responses_family.parameter_translation import TranslationContext, TranslationEnvelope
from ..responses_family.responses import ResponsesMapper
from ..responses_family.transport import HttpxClientConfig, build_httpx_client_config
from . import multimodal, tools
from .parameter_translation import apply_ark_responses_parameters

VOLCENGINE_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
VOLCENGINE_API_PREFIX = "api/v3"
VOLCENGINE_PROVIDER_LABEL = "Volcengine Ark"
ARK_RESPONSES_ENDPOINT = "responses"
ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT = "embeddings/multimodal"
ARK_DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=60.0)
ARK_CLIENT_REQUEST_ID_HEADER = "X-Client-Request-Id"
_ARK_EMBEDDING_CATALOG = get_parameter_catalog("volcengine_ark", "embeddings")
ARK_EMBEDDING_DIRECT_BODY_KEYS = set(_ARK_EMBEDDING_CATALOG.direct_body_keys)
ARK_EMBEDDING_RESERVED_BODY_KEYS = set(_ARK_EMBEDDING_CATALOG.reserved_body_keys)
ARK_BETA_TOOL_HEADERS = {
    "web_search": "ark-beta-web-search",
    "mcp": "ark-beta-mcp",
    "knowledge_search": "ark-beta-knowledge-search",
    "doubao_app": "ark-beta-doubao-app",
    "image_process": "ark-beta-image-process",
}


class ArkResponsesMapper(ResponsesMapper):
    """通过 ARK Provider 门面调用 Responses Family。"""

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
        return tools.convert_tools(tool_options)

    def _convert_user_content_parts(
        self,
        message: MessageSnapshot,
    ) -> list[OpenAIInputTextBlock | OpenAIInputImageBlock]:
        return multimodal.convert_user_content_parts(message, logger=self.logger, options=self.options)

    def _apply_response_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        apply_ark_responses_parameters(context, envelope)

    def _extract_tool_calls(self, output: list[OpenAIResponseOutputItem]) -> list[ProviderToolCall]:
        return tools.extract_tool_calls(output, options=self.options)

    def _extract_text_content(self, response_model: OpenAIResponseSnapshot) -> str:
        return multimodal.extract_text_content(response_model)

    def _extract_reasoning_content(self, output: list[OpenAIResponseOutputItem]) -> str | None:
        return multimodal.extract_reasoning_content(output)


def create_responses_mapper(*, options: ProviderRuntimeOptions, logger: logging.Logger) -> ResponsesMapper:
    return ArkResponsesMapper(
        options=options,
        logger=logger,
        provider_label=VOLCENGINE_PROVIDER_LABEL,
        raw_provider="volcengine_ark_responses",
        policy_provider="volcengine_ark",
    )


def build_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str,
    force_official_endpoint: bool,
    default_max_retries: int = 3,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    return build_httpx_client_config(
        api_provider,
        default_base_url=VOLCENGINE_ARK_BASE_URL,
        user_agent=user_agent,
        force_default_base_url=force_official_endpoint,
        default_timeout=ARK_DEFAULT_TIMEOUT,
        default_max_retries=default_max_retries,
        force_max_retries=force_max_retries,
        default_retry_interval=default_retry_interval,
        force_retry_interval=force_retry_interval,
    )


def _has_header(headers: Mapping[str, str], name: str) -> bool:
    normalized_name = name.lower()
    return any(key.lower() == normalized_name for key in headers)


def _ark_tool_type(tool: object) -> str | None:
    tool_mapping = json_mapping_or_none(tool)
    if tool_mapping is None:
        return None
    value = tool_mapping.get("type")
    return value if isinstance(value, str) else None


def build_ark_request_headers(headers: Mapping[str, str], body: dict) -> dict[str, str]:
    result = dict(headers)
    if not _has_header(result, ARK_CLIENT_REQUEST_ID_HEADER):
        result[ARK_CLIENT_REQUEST_ID_HEADER] = str(uuid.uuid4())
    tools = json_list_or_none(body.get("tools")) or []
    for tool in tools:
        beta_header = ARK_BETA_TOOL_HEADERS.get(_ark_tool_type(tool) or "")
        if beta_header is not None and not _has_header(result, beta_header):
            result[beta_header] = "true"
    return result
