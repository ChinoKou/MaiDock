import logging
import uuid
from collections.abc import Mapping

import httpx

from ...core.common import ArkBuiltinEndpointMode, RuntimeOptionsView
from ...core.json_types import JsonValue, json_list_or_none, json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...schemas import (
    ApiProviderSnapshot,
    MessageSnapshot,
    OpenAIEasyInputMessage,
    OpenAIInputImageBlock,
    OpenAIInputTextBlock,
    OpenAIResponseInputItem,
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
VOLCENGINE_ARK_AGENT_PLAN_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
VOLCENGINE_AGENT_PLAN_API_PREFIX = "api/plan/v3"
VOLCENGINE_ARK_CODING_PLAN_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
VOLCENGINE_CODING_PLAN_API_PREFIX = "api/coding/v3"
VOLCENGINE_PROVIDER_LABEL = "Volcengine Ark"
ARK_RESPONSES_ENDPOINT = "responses"
ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT = "embeddings/multimodal"
ARK_DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=60.0)
ARK_CLIENT_REQUEST_ID_HEADER = "X-Client-Request-Id"
_ARK_EMBEDDING_CATALOG = get_parameter_catalog("volcengine_ark", "embeddings")
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

    def _convert_messages(self, messages: list[MessageSnapshot]) -> list[OpenAIResponseInputItem]:
        """给末位 assistant 消息补上 ARK 必填的续写标记。

        ARK Responses API 规定：input 末条消息 role 为 assistant 即进入续写模式，
        且「在续写模式下，partial 为必填项」；缺失时上游直接 400 InvalidParameter，
        报错文本里就是那个 partial。Host 的 planner 恰好每轮都以 assistant 预填收尾，
        这里不打标它就一定失败。续写语义（响应只含续写部分）与 Anthropic 预填一致，
        Host 侧本就按此消费；OpenAI Responses 没有 partial 字段，所以只在 ARK 覆写。
        """
        converted = super()._convert_messages(messages)
        if converted:
            last = converted[-1]
            if isinstance(last, OpenAIEasyInputMessage):
                converted[-1] = last.model_copy(update={"partial": True})
        return converted

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


def create_responses_mapper(*, options: RuntimeOptionsView, logger: logging.Logger) -> ResponsesMapper:
    return ArkResponsesMapper(
        options=options,
        logger=logger,
        provider_label=VOLCENGINE_PROVIDER_LABEL,
        raw_provider="volcengine_ark_responses",
        policy_provider="volcengine_ark",
    )


def builtin_endpoint_profile(mode: ArkBuiltinEndpointMode) -> tuple[str, str]:
    """按内置端点类型返回 (default_base_url, api_prefix)。

    Agent Plan / Coding Plan 是订阅制专属端点（/api/plan/v3、/api/coding/v3，
    OpenAI 兼容且都有 /responses），与按量付费的 /api/v3 只差路径前缀，鉴权同为
    Bearer（Agent Plan 需其专属 API Key）。api_prefix 必须与 base_url 同步交给
    resolve_endpoint_path：前缀对不上时它识别不出 base 里已含前缀，会拼出
    /api/plan/v3/api/v3/responses 这样的双前缀路径。
    """

    if mode == "agent_plan":
        return VOLCENGINE_ARK_AGENT_PLAN_BASE_URL, VOLCENGINE_AGENT_PLAN_API_PREFIX
    if mode == "coding_plan":
        return VOLCENGINE_ARK_CODING_PLAN_BASE_URL, VOLCENGINE_CODING_PLAN_API_PREFIX
    return VOLCENGINE_ARK_BASE_URL, VOLCENGINE_API_PREFIX


def build_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str,
    force_official_endpoint: bool,
    builtin_endpoint_mode: ArkBuiltinEndpointMode = "standard",
    default_max_retries: int = 3,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    default_base_url, _ = builtin_endpoint_profile(builtin_endpoint_mode)
    return build_httpx_client_config(
        api_provider,
        default_base_url=default_base_url,
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


def build_ark_request_headers(headers: Mapping[str, str], body: dict[str, JsonValue]) -> dict[str, str]:
    result = dict(headers)
    if not _has_header(result, ARK_CLIENT_REQUEST_ID_HEADER):
        result[ARK_CLIENT_REQUEST_ID_HEADER] = str(uuid.uuid4())
    tools = json_list_or_none(body.get("tools")) or []
    for tool in tools:
        beta_header = ARK_BETA_TOOL_HEADERS.get(_ark_tool_type(tool) or "")
        if beta_header is not None and not _has_header(result, beta_header):
            result[beta_header] = "true"
    return result
