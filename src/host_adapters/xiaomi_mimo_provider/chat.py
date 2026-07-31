import logging

from ...core.common import (
    RuntimeOptionsView,
    read_api_key,
    read_timeout,
    resolve_max_retries,
    resolve_retry_interval,
)
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
    normalize_base_url,
    resolve_endpoint_path,
    with_default_user_agent,
)
from . import multimodal, tools
from .parameter_translation import apply_mimo_chat_parameters
from ...core.json_types import JsonValue

MIMO_PROVIDER_LABEL = "Xiaomi Mimo"
MIMO_CHAT_COMPLETIONS_ENDPOINT = "chat/completions"


class MimoChatCompletionsMapper(ChatCompletionsMapper):
    """使用 Mimo 专用字段映射的 Chat Completions mapper。"""

    def _convert_message_content(self, message: MessageSnapshot) -> str | list[dict[str, JsonValue]] | None:
        return multimodal.convert_message_content(message, options=self.options, logger=self.logger)

    def _build_image_content(self, part: MessagePartImage) -> dict[str, JsonValue] | None:
        return multimodal.build_image_content(part, options=self.options, logger=self.logger)

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[dict[str, JsonValue]]:
        return tools.convert_tools(tool_options)

    def _convert_history_tool_call(self, tool_call: ToolCallSnapshot, *, index: int = 1) -> dict[str, JsonValue] | None:
        return tools.convert_history_tool_call(tool_call, options=self.options, index=index)

    def _extract_tool_calls(self, raw_tool_calls: object) -> list[ProviderToolCall]:
        return tools.extract_tool_calls(raw_tool_calls, options=self.options)

    def _message_content_text(self, value: object) -> str | None:
        return multimodal.message_content_text(value)

    def _apply_chat_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        apply_mimo_chat_parameters(context, envelope)


def build_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str,
    default_max_retries: int = 3,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    api_key = read_api_key(api_provider)
    default_headers = {}
    if api_provider.default_headers:
        for key, value in api_provider.default_headers.fields.items():
            if isinstance(value, str):
                default_headers[str(key)] = value
    default_headers["api-key"] = api_key
    default_headers = with_default_user_agent(default_headers, user_agent)
    default_headers.setdefault("Accept", "application/json")
    default_headers.setdefault("Content-Type", "application/json")

    base_url = normalize_base_url(api_provider.base_url)

    default_query = api_provider.default_query.to_plain_dict()

    return HttpxClientConfig(
        base_url=base_url,
        default_headers=default_headers,
        default_query=default_query,
        timeout=read_timeout(api_provider),
        max_retries=resolve_max_retries(
            api_provider,
            config_value=default_max_retries,
            force=force_max_retries,
            default=3,
        ),
        retry_interval=resolve_retry_interval(
            api_provider,
            config_value=default_retry_interval,
            force=force_retry_interval,
            default=5.0,
        ),
    )


def resolve_path(config: HttpxClientConfig, endpoint: str) -> str:
    return resolve_endpoint_path(config.base_url, api_prefix="", endpoint_path=endpoint)


def _create_mapper(*, options: RuntimeOptionsView, logger: logging.Logger) -> ChatCompletionsMapper:
    return MimoChatCompletionsMapper(
        options=options,
        logger=logger,
        provider_label=MIMO_PROVIDER_LABEL,
        raw_provider="xiaomi_mimo",
        policy_provider="xiaomi_mimo",
        tool_namespace="xiaomi_mimo",
        history_tool_prefix="mimo_history_tool",
        extract_tool_prefix="mimo_tool",
    )


def build_chat_body(
    request: ResponseRequestSnapshot,
    *,
    options: RuntimeOptionsView,
    logger: logging.Logger,
    stream: bool,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue]]:
    mapper = _create_mapper(options=options, logger=logger)
    return mapper.build_request_body(request, stream=stream)


def convert_response(payload: dict[str, JsonValue], *, options: RuntimeOptionsView) -> ProviderResponse:
    mapper = _create_mapper(options=options, logger=logging.getLogger(__name__))
    return mapper.convert_response(payload)
