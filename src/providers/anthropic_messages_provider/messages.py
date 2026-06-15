import logging
from typing import Literal

from ...core.common import (
    ProviderRuntimeOptions,
    build_usage_from_snapshot,
    message_text,
    normalize_base_url,
    read_api_key,
    read_model_identifier,
    read_timeout,
    require_string_mapping,
    with_default_user_agent,
)
from ...core.diagnostics import build_parse_error_message, sanitize_json_object
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...providers.common.parameter_translation import (
    NormalizedHostParameters,
    TranslationContext,
    TranslationEnvelope,
    build_translation_context,
)
from ...schemas import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicRawData,
    AnthropicResponseSnapshot,
    AnthropicTextBlock,
    AnthropicToolResultBlock,
    ApiProviderSnapshot,
    MessageSnapshot,
    ProviderFunctionCall,
    ProviderResponse,
    ProviderToolCall,
    ResponseRequestSnapshot,
)
from ..common.httpx import HttpxClientConfig
from ..common.reasoning import merge_reasoning_and_xml_tool_fallback
from .multimodal import convert_content_blocks
from .parameter_translation import apply_anthropic_parameters
from .tools import convert_assistant_tool_calls, convert_tools, orphan_tool_result_message

ANTHROPIC_PROVIDER_LABEL = "Anthropic Messages"
ANTHROPIC_API_PREFIX = "v1"
ANTHROPIC_MESSAGES_ENDPOINT = "messages"
ANTHROPIC_VERSION = "2023-06-01"


def _auth_header_value(prefix: str, api_key: str) -> str:
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        return api_key
    return f"{normalized_prefix} {api_key}"


def build_client_config(api_provider: ApiProviderSnapshot, *, user_agent: str) -> HttpxClientConfig:
    default_headers = require_string_mapping(api_provider.default_headers, field_name="api_provider.default_headers")
    default_query = api_provider.default_query.to_plain_dict()
    auth_type = (api_provider.auth_type or "bearer").strip().lower()
    api_key = read_api_key(api_provider, allow_empty=auth_type == "none")

    if auth_type == "bearer":
        lowered_headers = {key.lower() for key in default_headers}
        if "x-api-key" not in lowered_headers and "authorization" not in lowered_headers:
            default_headers["x-api-key"] = api_key
    elif auth_type == "header":
        header_name = api_provider.auth_header_name
        if header_name.lower() == "x-api-key":
            default_headers[header_name] = api_key
        else:
            default_headers[header_name] = _auth_header_value(api_provider.auth_header_prefix, api_key)
    elif auth_type == "query":
        default_query[api_provider.auth_query_name] = api_key
    elif auth_type != "none":
        raise ValueError(f"不支持的 auth_type: {api_provider.auth_type}")

    normalized_base_url = normalize_base_url(api_provider.base_url)
    headers = with_default_user_agent(default_headers, user_agent)
    headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
    headers.setdefault("Accept", "application/json")
    headers.setdefault("Content-Type", "application/json")
    return HttpxClientConfig(
        base_url=normalized_base_url,
        default_headers=headers,
        default_query=default_query,
        timeout=read_timeout(api_provider),
    )


def build_request(
    request: ResponseRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
) -> AnthropicMessagesRequest:
    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get("anthropic_messages", "chat_completion")
    catalog = get_parameter_catalog("anthropic_messages", "chat_completion")

    normalized = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=ANTHROPIC_PROVIDER_LABEL,
        provider="anthropic_messages",
        capability="chat_completion",
        model=model,
    ).normalized

    tools = convert_tools(request.tool_options)
    default_tool_choice = {"type": "any", "disable_parallel_tool_use": True}
    is_tool_required: bool = False
    if tools and "tool_choice" not in normalized.fields:
        normalized.fields["tool_choice"] = default_tool_choice
        normalized.sources["tool_choice"] = "provider.default_tool_choice"
        is_tool_required = True

    return AnthropicMessagesRequest(
        model=model,
        messages=convert_messages(request.message_list, options=options, logger=logger),
        max_tokens=0,
        system=extract_system(request.message_list, is_tool_required=is_tool_required),
        temperature=None,
        tools=tools,
        normalized=normalized,
    )


def build_http_body(
    upstream_request: AnthropicMessagesRequest,
    *,
    options: ProviderRuntimeOptions,
    stream: bool,
) -> dict:
    normalized = upstream_request.normalized
    if not isinstance(normalized, NormalizedHostParameters):
        raise TypeError("AnthropicMessagesRequest.normalized 缺少 NormalizedHostParameters")

    policy = options.parameter_policies.get("anthropic_messages", "chat_completion")
    catalog = get_parameter_catalog("anthropic_messages", "chat_completion")

    messages = upstream_request.message_params()
    tool_params = upstream_request.tool_params()
    body = {"model": upstream_request.model, "messages": messages, "stream": stream}

    system_value = upstream_request.system
    if system_value is not None and "system" not in normalized.fields:
        body["system"] = system_value

    if tool_params and "tools" not in normalized.fields:
        body["tools"] = tool_params

    envelope = TranslationEnvelope(body=body)

    context_obj = TranslationContext(
        request=None,
        provider_label=ANTHROPIC_PROVIDER_LABEL,
        provider="anthropic_messages",
        capability="chat_completion",
        catalog=catalog,
        policy=policy,
        normalized=normalized,
        model=upstream_request.model,
    )
    apply_anthropic_parameters(context_obj, envelope)

    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers={**upstream_request.extra_headers, **envelope.headers},
        query={**upstream_request.extra_query, **envelope.query},
        policy=policy,
        provider_label=ANTHROPIC_PROVIDER_LABEL,
        capability="chat_completion",
    )
    upstream_request.extra_headers.clear()
    upstream_request.extra_headers.update(transport.headers)
    upstream_request.extra_query.clear()
    upstream_request.extra_query.update(transport.query)
    return transport.body


def extract_system(messages: list[MessageSnapshot], *, is_tool_required: bool = False) -> str | None:
    del is_tool_required
    system = "\n\n".join(
        message_text(message) for message in messages if message.role == "system" and message_text(message)
    )
    return system or None


def convert_messages(
    messages: list[MessageSnapshot],
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
) -> list[AnthropicMessage]:
    converted: list[AnthropicMessage] = []
    emitted_tool_use_ids: set[str] = set()
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            tool_use_id = (message.tool_call_id or "").strip()
            if tool_use_id and tool_use_id in emitted_tool_use_ids:
                converted.append(
                    AnthropicMessage(
                        role="user",
                        content=[AnthropicToolResultBlock(tool_use_id=tool_use_id, content=message_text(message))],
                    )
                )
            else:
                converted.append(orphan_tool_result_message(message))
            continue
        if message.role not in {"user", "assistant"}:
            continue
        content = convert_content_blocks(message, options=options, logger=logger)
        if message.role == "assistant":
            tool_use_blocks = convert_assistant_tool_calls(message, options=options)
            content.extend(tool_use_blocks)
            emitted_tool_use_ids.update(block.id for block in tool_use_blocks)
        if content:
            role: Literal["user", "assistant"] = "assistant" if message.role == "assistant" else "user"
            converted.append(AnthropicMessage(role=role, content=content))
    if not converted:
        converted.append(AnthropicMessage(role="user", content=[AnthropicTextBlock(text="")]))
    return converted


def convert_response(response: object, *, options: ProviderRuntimeOptions) -> ProviderResponse:
    response_model = AnthropicResponseSnapshot.model_validate(response)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ProviderToolCall] = []
    for block in response_model.content:
        if block.type == "text" and block.text:
            text_parts.append(block.text)
        elif block.type in {"thinking", "redacted_thinking"}:
            thinking = block.thinking or block.text
            if thinking:
                reasoning_parts.append(thinking)
        elif block.type == "tool_use":
            tool_calls.append(
                ProviderToolCall(
                    id=block.id or "",
                    function=ProviderFunctionCall(name=block.name or "", arguments=block.input),
                    extra_content={"provider": "anthropic_messages"},
                )
            )

    content = "".join(text_parts) or None
    native_reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    reasoning_content, final_content = merge_reasoning_and_xml_tool_fallback(
        content=content,
        native_reasoning=native_reasoning,
        tool_calls=tool_calls,
        options=options,
    )

    usage = build_usage_from_snapshot(response_model.usage)
    raw_data = (
        AnthropicRawData(
            id=response_model.id,
            model=response_model.model,
            stop_reason=response_model.stop_reason,
            usage=response_model.usage,
        ).to_host_dict()
        if options.include_raw_data
        else None
    )
    if not final_content and not tool_calls:
        raise ValueError(build_parse_error_message(ANTHROPIC_PROVIDER_LABEL, "响应中既没有文本内容，也没有工具调用"))
    return ProviderResponse(
        content=final_content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        usage=usage,
        raw_data=sanitize_json_object(raw_data) if raw_data is not None else None,
    )
