import logging
import re
from collections.abc import Mapping
from typing import Literal

from ...core.common import (
    RuntimeOptionsView,
    build_usage_from_snapshot,
    image_data_url,
    message_text,
    read_model_identifier,
)
from ...core.diagnostics import build_parse_error_message
from ...core.json_types import (
    JsonValue,
    json_array,
    json_mapping_or_none,
    list_field,
    mapping_field,
    mapping_to_json_object,
)
from ...core.parameter_catalog import get_parameter_catalog
from ...i18n import runtime_item, translate
from ...schemas import (
    ApiProviderSnapshot,
    GenericUsageSnapshot,
    MessagePartImage,
    MessagePartText,
    MessageSnapshot,
    ProviderResponse,
    ProviderToolCall,
    ResponseRequestSnapshot,
)
from ..common.httpx import HttpxClientConfig, HttpxProviderParseError, build_httpx_client_config, resolve_endpoint_path
from ..common.parameter_translation import TranslationEnvelope, build_translation_context
from ..common.payloads import raw_data_or_none
from ..common.reasoning import merge_reasoning_and_xml_tool_fallback
from .errors import raise_for_dashscope_error
from .parameter_translation import apply_dashscope_chat_parameters, normalize_dashscope_chat_body
from .tools import convert_history_tool_call, convert_tools, extract_tool_calls

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_PREFIX = "api/v1"
DASHSCOPE_PROVIDER_LABEL = "阿里云百炼 DashScope"
DASHSCOPE_GENERATION_ENDPOINT = "services/aigc/text-generation/generation"
DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT = "services/aigc/multimodal-generation/generation"
DASHSCOPE_DEFAULT_TIMEOUT = 300.0
type DashScopeEndpointKind = Literal["text", "multimodal"]

_DASHSCOPE_TEXT_ONLY_MODELS = frozenset(
    {
        "qwen3.7-max",
        "qwen3.7-max-2026-05-20",
        "qwen3.6-max-preview",
    }
)
_DASHSCOPE_EXPLICIT_MULTIMODAL_MODELS = frozenset({"qwen3.7-max-2026-06-08"})
_DASHSCOPE_MULTIMODAL_TOKEN_PATTERN = re.compile(r"(?:^|[-_/])(vl|audio|omni)(?:$|[-_/])", re.IGNORECASE)


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
        default_base_url=DASHSCOPE_BASE_URL,
        user_agent=user_agent,
        force_default_base_url=force_official_endpoint,
        default_timeout=DASHSCOPE_DEFAULT_TIMEOUT,
        default_max_retries=default_max_retries,
        force_max_retries=force_max_retries,
        default_retry_interval=default_retry_interval,
        force_retry_interval=force_retry_interval,
    )


def resolve_path(config: HttpxClientConfig, endpoint: str) -> str:
    return resolve_endpoint_path(config.base_url, api_prefix=DASHSCOPE_API_PREFIX, endpoint_path=endpoint)


def normalize_dashscope_model(model: str) -> str:
    return model.strip().lower()


def is_dashscope_text_only_model(model: str) -> bool:
    return normalize_dashscope_model(model) in _DASHSCOPE_TEXT_ONLY_MODELS


def is_dashscope_multimodal_model(model: str) -> bool:
    normalized = normalize_dashscope_model(model)
    if normalized in _DASHSCOPE_EXPLICIT_MULTIMODAL_MODELS:
        return True
    if normalized.startswith(("qwen3.7-plus", "qwen3.6-", "qwen3.5-", "qvq")):
        return normalized not in _DASHSCOPE_TEXT_ONLY_MODELS
    return _DASHSCOPE_MULTIMODAL_TOKEN_PATTERN.search(normalized) is not None


def dashscope_endpoint_path(config: HttpxClientConfig, kind: DashScopeEndpointKind) -> str:
    endpoint = DASHSCOPE_GENERATION_ENDPOINT if kind == "text" else DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT
    return resolve_path(config, endpoint)


def request_has_image(messages: list[MessageSnapshot]) -> bool:
    return any(message.role == "user" and _has_image_part(message) for message in messages)


def first_choice(payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue] | None:
    output = mapping_field(payload, "output")
    if output is None:
        return None
    choices = list_field(output, "choices")
    if not choices:
        return None
    return json_mapping_or_none(choices[0])


def first_choice_message(
    payload: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue] | None:
    choice = first_choice(payload)
    if choice is None:
        return None
    return mapping_field(choice, "message")


def string_value(mapping: Mapping[str, JsonValue], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def is_multimodal_endpoint(path: str) -> bool:
    return DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT.rstrip("/") in path


def extract_content_text(content: object, *, is_multimodal: bool) -> str | None:
    """从 message.content 中提取文本，兼容多模态端点与文本端点的不同格式。"""
    if isinstance(content, str):
        return content or None
    if is_multimodal:
        if isinstance(content, list):
            parts = [
                item["text"]
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"]
            ]
            return "\n".join(parts) if parts else None
        return None
    return None


def extract_reasoning_from_mapping(mapping: Mapping[str, JsonValue]) -> str | None:
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def content_message(role: str, content: str | list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    return {"role": role, "content": content if isinstance(content, str) else json_array(content)}


def build_generation_body(
    request: ResponseRequestSnapshot,
    *,
    options: RuntimeOptionsView,
    stream: bool,
    logger: logging.Logger,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue]]:
    model = read_model_identifier(request.model_info)
    overrides = options.parameter_overrides.get("dashscope", "chat_completion")
    catalog = get_parameter_catalog("dashscope", "chat_completion")

    context = build_translation_context(
        request,
        overrides=overrides,
        catalog=catalog,
        provider_label=DASHSCOPE_PROVIDER_LABEL,
        provider="dashscope",
        capability="chat_completion",
        model=model,
    )

    messages = convert_messages(request.message_list, options=options, logger=logger)
    tools = convert_tools(request.tool_options)

    parameters: dict[str, JsonValue] = {"result_format": "message"}
    if tools:
        parameters["tools"] = json_array(tools)
    if stream and "stream" not in context.normalized.fields:
        if "incremental_output" not in context.normalized.fields:
            parameters["incremental_output"] = True
        parameters["stream"] = True

    body: dict[str, JsonValue] = {
        "model": model,
        "input": {"messages": json_array(messages)},
        "parameters": parameters,
    }
    envelope = TranslationEnvelope(body=body)
    apply_dashscope_chat_parameters(context, envelope)

    normalized_body = dict(envelope.body)
    normalize_dashscope_chat_body(normalized_body, context=context)
    return normalized_body, envelope.headers, envelope.query


def count_tools(body: dict[str, JsonValue]) -> int:
    tools = mapping_field(body, "parameters")
    if tools is None:
        return 0
    tools_list = list_field(tools, "tools")
    return len(tools_list) if tools_list is not None else 0


def convert_messages(
    messages: list[MessageSnapshot],
    *,
    options: RuntimeOptionsView,
    logger: logging.Logger,
) -> list[dict[str, JsonValue]]:
    converted: list[dict[str, JsonValue]] = []
    for message in messages:
        if message.role not in {"system", "user", "assistant", "tool"}:
            continue
        if message.role == "tool":
            tool_message = content_message("tool", message_text(message))
            if message.tool_call_id:
                tool_message["tool_call_id"] = message.tool_call_id
            if message.tool_name:
                tool_message["name"] = message.tool_name
            converted.append(tool_message)
            continue

        if message.role == "user" and _has_image_part(message):
            current = {
                "role": message.role,
                "content": _convert_multimodal_content(message, options=options, logger=logger),
            }
        else:
            current = content_message(message.role, message_text(message))

        if message.role == "assistant":
            tool_calls = [convert_history_tool_call(tool_call, options=options) for tool_call in message.tool_calls]
            filtered_tool_calls = [tool_call for tool_call in tool_calls if tool_call is not None]
            if filtered_tool_calls:
                current["tool_calls"] = json_array(filtered_tool_calls)
        if current.get("content") or current.get("tool_calls"):
            converted.append(current)
    return converted


def _has_image_part(message: MessageSnapshot) -> bool:
    return any(isinstance(part, MessagePartImage) for part in message.parts)


def _convert_multimodal_content(
    message: MessageSnapshot,
    *,
    options: RuntimeOptionsView,
    logger: logging.Logger,
) -> list[dict[str, JsonValue]]:
    content_blocks: list[dict[str, JsonValue]] = []
    for part in message.parts:
        if isinstance(part, MessagePartText) and part.text:
            content_blocks.append({"text": part.text})
        elif isinstance(part, MessagePartImage):
            data_url = image_data_url(part, logger, options.invalid_image_policy, options.image_limits)
            if data_url:
                content_blocks.append({"image": data_url})
            elif options.invalid_image_policy == "placeholder":
                content_blocks.append({"text": "[图片内容不可用]"})
    return content_blocks


def dashscope_usage(payload: Mapping[str, JsonValue]) -> GenericUsageSnapshot:
    usage = mapping_field(payload, "usage")
    if usage is not None:
        return GenericUsageSnapshot.model_validate(mapping_to_json_object(usage))
    return GenericUsageSnapshot()


def convert_response(
    payload: dict[str, JsonValue], *, options: RuntimeOptionsView, is_multimodal: bool = False
) -> ProviderResponse:
    raise_for_dashscope_error(payload)
    payload_mapping = payload
    message = first_choice_message(payload_mapping)
    content: str | None = None
    native_reasoning: str | None = None
    tool_calls: list[ProviderToolCall] = []
    if message is not None:
        content = extract_content_text(message.get("content"), is_multimodal=is_multimodal)
        native_reasoning = extract_reasoning_from_mapping(message)
        tool_calls = extract_tool_calls(message.get("tool_calls"), options=options)
    if content is None:
        output = mapping_field(payload_mapping, "output")
        if output is not None:
            content = string_value(output, "text")
            native_reasoning = native_reasoning or extract_reasoning_from_mapping(output)
    reasoning_content, final_content = merge_reasoning_and_xml_tool_fallback(
        content=content,
        native_reasoning=native_reasoning,
        tool_calls=tool_calls,
        options=options,
    )
    if not final_content and not tool_calls:
        missing_message = translate(
            "runtime.error.output_missing",
            item=runtime_item("output_text_or_tools"),
        )
        raise HttpxProviderParseError(build_parse_error_message(DASHSCOPE_PROVIDER_LABEL, missing_message))
    return ProviderResponse(
        content=final_content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        usage=build_usage_from_snapshot(dashscope_usage(payload_mapping)),
        raw_data=raw_data_or_none(payload, options=options),
    )
