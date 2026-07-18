import logging
import re

from ...core.common import ProviderRuntimeOptions
from ...schemas import (
    ApiProviderSnapshot,
    MessagePartImage,
    MessageSnapshot,
    ProviderResponse,
    ProviderToolCall,
    ResponseRequestSnapshot,
    ToolCallSnapshot,
    ToolOptionSnapshot,
)
from ..chat_completions_family.chat import ChatCompletionsMapper
from ..chat_completions_family.parameter_translation import TranslationContext, TranslationEnvelope
from ..chat_completions_family.transport import (
    HttpxClientConfig,
    build_httpx_client_config,
    resolve_endpoint_path,
)
from . import multimodal, tools
from .parameter_translation import apply_siliconflow_chat_parameters

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
    default_max_retries: int = 3,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    return build_httpx_client_config(
        api_provider,
        default_base_url=SILICONFLOW_BASE_URL,
        user_agent=user_agent,
        force_default_base_url=force_official_endpoint,
        default_max_retries=default_max_retries,
        force_max_retries=force_max_retries,
        default_retry_interval=default_retry_interval,
        force_retry_interval=force_retry_interval,
    )


def resolve_path(config: HttpxClientConfig, endpoint: str) -> str:
    return resolve_endpoint_path(config.base_url, api_prefix=SILICONFLOW_API_PREFIX, endpoint_path=endpoint)


class SiliconFlowChatCompletionsMapper(ChatCompletionsMapper):
    """通过 SiliconFlow 适配门面调用 Chat Completions Family。"""

    def _convert_message_content(self, message: MessageSnapshot) -> str | list[dict] | None:
        return multimodal.convert_message_content(message, options=self.options, logger=self.logger)

    def _build_image_content(self, part: MessagePartImage) -> dict | None:
        return multimodal.build_image_content(part, options=self.options, logger=self.logger)

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[dict]:
        return tools.convert_tools(tool_options)

    def _convert_history_tool_call(self, tool_call: ToolCallSnapshot, *, index: int = 1) -> dict | None:
        return tools.convert_history_tool_call(tool_call, options=self.options, index=index)

    def _extract_tool_calls(self, raw_tool_calls: object) -> list[ProviderToolCall]:
        return tools.extract_tool_calls(raw_tool_calls, options=self.options)

    def _message_content_text(self, value: object) -> str | None:
        return multimodal.message_content_text(value)

    def _apply_chat_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        apply_siliconflow_chat_parameters(context, envelope)


def _create_mapper(*, options: ProviderRuntimeOptions, logger: logging.Logger) -> ChatCompletionsMapper:
    return SiliconFlowChatCompletionsMapper(
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
