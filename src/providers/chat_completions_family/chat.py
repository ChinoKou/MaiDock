import logging
from collections.abc import Mapping

from ...core.common import (
    ProviderRuntimeOptions,
    build_usage_from_snapshot,
    message_text,
    read_model_identifier,
)
from ...core.diagnostics import build_parse_error_message
from ...core.json_types import JsonValue, json_mapping_or_none, list_field, mapping_field
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import ProviderPolicyKey, apply_transport_parameter_policy
from ...schemas import (
    GenericUsageSnapshot,
    MessagePartImage,
    MessageSnapshot,
    ProviderResponse,
    ProviderToolCall,
    ProviderUsage,
    ResponseRequestSnapshot,
    ToolCallSnapshot,
    ToolOptionSnapshot,
)
from ..common.httpx import HttpxProviderParseError
from ..common.parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    build_translation_context,
)
from ..common.payloads import raw_data_or_none
from ..common.reasoning import merge_reasoning_and_xml_tool_fallback
from .multimodal import (
    build_image_content as build_family_image_content,
)
from .multimodal import (
    convert_message_content as convert_family_message_content,
)
from .multimodal import (
    message_content_text as family_message_content_text,
)
from .parameter_translation import apply_chat_completions_family_parameters
from .tools import (
    convert_history_tool_call as convert_family_history_tool_call,
)
from .tools import (
    convert_tools as convert_family_tools,
)
from .tools import (
    extract_tool_calls as extract_family_tool_calls,
)


def _string_value(mapping: Mapping[str, JsonValue], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def _first_choice(payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue] | None:
    choices = list_field(payload, "choices")
    if not choices:
        return None
    return json_mapping_or_none(choices[0])


def _first_choice_message(
    payload: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue] | None:
    choice = _first_choice(payload)
    if choice is None:
        return None
    return mapping_field(choice, "message")


def _message_content_text(value: object) -> str | None:
    """从消息 content 字段中提取纯文本内容。"""
    return family_message_content_text(value)


class ChatCompletionsMapper:
    """构建并解析 OpenAI Chat Completions 兼容的请求/响应负载。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        logger: logging.Logger,
        provider_label: str,
        raw_provider: str,
        policy_provider: ProviderPolicyKey,
        tool_namespace: str = "chat_completions",
        history_tool_prefix: str = "chat_completions_history_tool",
        extract_tool_prefix: str = "chat_completions_tool",
    ) -> None:
        self.options = options
        self.logger = logger
        self.provider_label = provider_label
        self.raw_provider = raw_provider
        self.policy_provider: ProviderPolicyKey = policy_provider
        self.tool_namespace = tool_namespace
        self.history_tool_prefix = history_tool_prefix
        self.extract_tool_prefix = extract_tool_prefix

    # ------------------------------------------------------------------
    # 请求构建
    # ------------------------------------------------------------------

    def build_request_body(
        self,
        request: ResponseRequestSnapshot,
        *,
        stream: bool,
        apply_policy: bool = True,
    ) -> tuple[dict, dict[str, str], dict]:
        """构建 Chat Completions HTTP body + extra headers + extra query。"""
        model = read_model_identifier(request.model_info)
        policy = self.options.parameter_policies.get(self.policy_provider, "chat_completion")
        catalog = get_parameter_catalog(self.policy_provider, "chat_completion")

        context = build_translation_context(
            request,
            policy=policy,
            catalog=catalog,
            provider_label=self.provider_label,
            provider=self.policy_provider,
            capability="chat_completion",
            model=model,
        )

        messages = self._convert_messages(request)
        tools = self._convert_tools(request.tool_options)

        body: dict = {"model": model, "messages": messages, "stream": stream}
        if tools and "tools" not in context.normalized.fields:
            body["tools"] = tools

        envelope = TranslationEnvelope(body=body)
        self._apply_chat_parameters(context, envelope)

        if not apply_policy:
            return envelope.body, envelope.headers, envelope.query

        transport = apply_transport_parameter_policy(
            body=envelope.body,
            headers=envelope.headers,
            query=envelope.query,
            policy=policy,
            provider_label=self.provider_label,
            capability="chat_completion",
        )
        return transport.body, transport.headers, transport.query

    # ------------------------------------------------------------------
    # 消息转换
    # ------------------------------------------------------------------

    def _convert_messages(self, request: ResponseRequestSnapshot) -> list[dict]:
        converted: list[dict] = []
        for message in request.message_list:
            if message.role not in {"system", "user", "assistant", "tool"}:
                continue
            if message.role == "tool":
                tool_message: dict = {"role": "tool", "content": message_text(message)}
                if message.tool_call_id:
                    tool_message["tool_call_id"] = message.tool_call_id
                if message.tool_name:
                    tool_message["name"] = message.tool_name
                converted.append(tool_message)
                continue

            content = self._convert_message_content(message)
            current: dict = {"role": message.role, "content": content}
            if message.role == "assistant":
                tool_calls = [
                    self._convert_history_tool_call(tc, index=i) for i, tc in enumerate(message.tool_calls, start=1)
                ]
                filtered_tool_calls = [tc for tc in tool_calls if tc is not None]
                if filtered_tool_calls:
                    current["tool_calls"] = filtered_tool_calls
            if content is not None or current.get("tool_calls"):
                converted.append(current)
        return converted

    def _convert_message_content(self, message: MessageSnapshot) -> str | list[dict] | None:
        return convert_family_message_content(
            message,
            options=self.options,
            logger=self.logger,
            image_builder=self._build_image_content,
        )

    def _build_image_content(self, part: MessagePartImage) -> dict | None:
        return build_family_image_content(part, logger=self.logger, options=self.options)

    # ------------------------------------------------------------------
    # 工具转换
    # ------------------------------------------------------------------

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[dict]:
        return convert_family_tools(tool_options)

    def _convert_history_tool_call(self, tool_call: ToolCallSnapshot, *, index: int = 1) -> dict | None:
        return convert_family_history_tool_call(
            tool_call,
            options=self.options,
            fallback_prefix=self.history_tool_prefix,
            index=index,
        )

    def _apply_chat_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        apply_chat_completions_family_parameters(context, envelope)

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    def convert_response(self, payload: dict) -> ProviderResponse:
        """解析非流式 Chat Completions 响应。"""
        message = _first_choice_message(payload)
        content: str | None = None
        native_reasoning: str | None = None
        tool_calls: list[ProviderToolCall] = []
        if message is not None:
            content = self._message_content_text(message.get("content"))
            native_reasoning = _string_value(message, "reasoning_content")
            tool_calls = self._extract_tool_calls(message.get("tool_calls"))

        reasoning_content, final_content = merge_reasoning_and_xml_tool_fallback(
            content=content,
            native_reasoning=native_reasoning,
            tool_calls=tool_calls,
            options=self.options,
        )
        if not final_content and not tool_calls:
            raise HttpxProviderParseError(
                build_parse_error_message(self.provider_label, "响应中既没有文本内容，也没有工具调用")
            )
        return ProviderResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=self._extract_usage(payload),
            raw_data=raw_data_or_none(payload, options=self.options),
        )

    def _extract_tool_calls(self, raw_tool_calls: object) -> list[ProviderToolCall]:
        return extract_family_tool_calls(
            raw_tool_calls,
            options=self.options,
            provider=self.raw_provider,
            namespace=self.tool_namespace,
            fallback_prefix=self.extract_tool_prefix,
        )

    def _message_content_text(self, value: object) -> str | None:
        return _message_content_text(value)

    @staticmethod
    def _extract_usage(payload: dict) -> ProviderUsage:
        return build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {}))
