import base64
import io
import json
import logging

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from src.core.common import InvalidImagePolicy, ProviderRuntimeOptions
from src.core.json_types import JsonValue
from src.providers.anthropic_messages_provider.messages import (
    build_client_config,
    build_http_body,
    build_request,
    convert_messages,
    convert_response,
    extract_system,
)
from src.providers.anthropic_messages_provider.multimodal import (
    anthropic_image_media_type,
    convert_content_blocks,
    convert_image_block,
)
from src.providers.anthropic_messages_provider.parameter_translation import (
    reject_anthropic_response_format_params,
)
from src.providers.anthropic_messages_provider.provider import AnthropicMessagesProvider
from src.providers.anthropic_messages_provider.tools import (
    build_default_tool_choice,
    convert_assistant_tool_calls,
    convert_tools,
    orphan_tool_result_message,
)
from src.schemas import (
    AnthropicMessagesRequest,
    ApiProviderSnapshot,
    MessagePartImage,
    MessageSnapshot,
    ObjectFields,
    ResponseRequestSnapshot,
    ToolOptionSnapshot,
)

LOGGER = logging.getLogger(__name__)
type JsonObject = dict[str, JsonValue]


def _api_provider(**overrides: JsonValue) -> ApiProviderSnapshot:
    payload: dict = {
        "api_key": "test-key",
        "auth_type": "bearer",
        "base_url": "https://example.com/v1",
        "default_headers": {},
        "default_query": {},
    }
    payload.update(overrides)
    return ApiProviderSnapshot.model_validate(payload)


def _response_snapshot(
    *,
    messages: list[JsonObject] | None = None,
    tools: list[JsonObject] | None = None,
    model_extra: JsonObject | None = None,
    request_extra: JsonObject | None = None,
    response_format: JsonObject | None = None,
    stream: bool = False,
    temperature: int | float | None = None,
    max_tokens: int | None = None,
) -> ResponseRequestSnapshot:
    payload: dict = {
        "model_info": {
            "model_identifier": "claude-test",
            "force_stream_mode": stream,
            "extra_params": model_extra if model_extra is not None else {},
        },
        "api_provider": {
            "api_key": "test-key",
            "auth_type": "bearer",
            "base_url": "https://example.com/v1",
        },
        "message_list": messages if messages is not None else [],
        "tool_options": tools if tools is not None else [],
        "extra_params": request_extra if request_extra is not None else {},
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return ResponseRequestSnapshot.model_validate(payload)


def _tool_definition(name: str = "lookup", *, nested: bool = True) -> JsonObject:
    function: JsonObject = {
        "name": name,
        "description": "查询信息",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        },
    }
    if nested:
        return {"type": "function", "function": function}
    return function


def _image_base64(image_format: str) -> str:
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _provider_request_payload(request: ResponseRequestSnapshot) -> dict:
    payload = request.model_dump(mode="json")
    payload["model_info"]["extra_params"] = {}
    payload["api_provider"]["default_headers"] = {}
    payload["api_provider"]["default_query"] = {}
    payload["extra_params"] = {}
    return payload


@pytest.mark.parametrize(
    ("overrides", "expected_header", "expected_value", "expected_query"),
    [
        pytest.param({}, "x-api-key", "test-key", {}, id="bearer-adds-x-api-key"),
        pytest.param(
            {"default_headers": {"X-API-Key": "existing"}},
            "X-API-Key",
            "existing",
            {},
            id="bearer-preserves-x-api-key",
        ),
        pytest.param(
            {"default_headers": {"Authorization": "Bearer existing"}},
            "Authorization",
            "Bearer existing",
            {},
            id="bearer-preserves-authorization",
        ),
        pytest.param(
            {"auth_type": "header", "auth_header_name": "x-api-key", "auth_header_prefix": "Ignored"},
            "x-api-key",
            "test-key",
            {},
            id="header-x-api-key-without-prefix",
        ),
        pytest.param(
            {"auth_type": "header", "auth_header_name": "X-Token", "auth_header_prefix": "Token"},
            "X-Token",
            "Token test-key",
            {},
            id="header-custom-prefix",
        ),
        pytest.param(
            {"auth_type": "header", "auth_header_name": "X-Token", "auth_header_prefix": "  "},
            "X-Token",
            "test-key",
            {},
            id="header-empty-prefix",
        ),
        pytest.param(
            {"auth_type": "query", "auth_query_name": "key", "default_query": {"region": "cn"}},
            None,
            None,
            {"region": "cn", "key": "test-key"},
            id="query",
        ),
        pytest.param(
            {"auth_type": "none", "api_key": ""},
            None,
            None,
            {},
            id="none-allows-empty-key",
        ),
    ],
)
def test_build_client_config_authentication_matrix(
    overrides: JsonObject,
    expected_header: str | None,
    expected_value: str | None,
    expected_query: JsonObject,
) -> None:
    config = build_client_config(_api_provider(**overrides), user_agent="Anthropic-UA/1")

    if expected_header is not None:
        assert config.default_headers[expected_header] == expected_value
    else:
        assert "x-api-key" not in {key.lower() for key in config.default_headers}
        assert "authorization" not in {key.lower() for key in config.default_headers}
    assert config.default_query == expected_query


def test_build_client_config_preserves_protocol_headers_and_retry_settings() -> None:
    api_provider = _api_provider(
        default_headers={
            "User-Agent": "Existing-UA/1",
            "anthropic-version": "2099-01-01",
            "Accept": "application/custom",
            "Content-Type": "application/custom+json",
        },
        timeout=12,
        max_retry=8,
        retry_interval=9,
    )

    host_config = build_client_config(api_provider, user_agent="Ignored-UA/1")
    forced_config = build_client_config(
        api_provider,
        user_agent="Ignored-UA/1",
        default_max_retries=2,
        force_max_retries=True,
        default_retry_interval=0.25,
        force_retry_interval=True,
    )

    assert host_config.default_headers == {
        "User-Agent": "Existing-UA/1",
        "anthropic-version": "2099-01-01",
        "Accept": "application/custom",
        "Content-Type": "application/custom+json",
        "x-api-key": "test-key",
    }
    assert host_config.timeout == 12.0
    assert host_config.max_retries == 8
    assert host_config.retry_interval == 9.0
    assert forced_config.max_retries == 2
    assert forced_config.retry_interval == 0.25


def test_extract_system_joins_nonempty_system_messages() -> None:
    messages = [
        MessageSnapshot.model_validate({"role": "system", "parts": [{"type": "text", "text": "第一段"}]}),
        MessageSnapshot.model_validate({"role": "system", "parts": [{"type": "text", "text": ""}]}),
        MessageSnapshot.model_validate({"role": "user", "parts": [{"type": "text", "text": "忽略"}]}),
        MessageSnapshot.model_validate({"role": "system", "parts": [{"type": "text", "text": "第二段"}]}),
    ]

    assert extract_system(messages) == "第一段\n\n第二段"
    assert extract_system([]) is None


def test_build_request_and_body_maps_all_anthropic_parameters_and_transport_roots() -> None:
    request = _response_snapshot(
        messages=[
            {"role": "system", "parts": [{"type": "text", "text": "系统一"}]},
            {"role": "system", "parts": [{"type": "text", "text": "系统二"}]},
            {"role": "user", "parts": [{"type": "text", "text": "问题"}]},
        ],
        tools=[_tool_definition()],
        request_extra={
            "top_p": 0.8,
            "top_k": 20,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"type": "tool", "name": "lookup", "disable_parallel_tool_use": False},
            "stop_sequences": ["STOP"],
            "metadata": {"user_id": "u-1"},
            "service_tier": "auto",
            "headers": {"anthropic-beta": "tools-2024"},
            "query": {"trace": True, "sample": 2},
        },
        temperature=0.4,
        max_tokens=512,
    )
    options = ProviderRuntimeOptions()

    upstream = build_request(request, options=options, logger=LOGGER)
    body = build_http_body(upstream, options=options, stream=True)

    assert body == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "问题"}]}],
        "stream": True,
        "system": "系统一\n\n系统二",
        "tools": [
            {
                "name": "lookup",
                "description": "查询信息",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
        "temperature": 0.4,
        "max_tokens": 512,
        "top_p": 0.8,
        "top_k": 20,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tool_choice": {"type": "tool", "name": "lookup", "disable_parallel_tool_use": False},
        "stop_sequences": ["STOP"],
        "metadata": {"user_id": "u-1"},
        "service_tier": "auto",
    }
    assert upstream.extra_headers == {"anthropic-beta": "tools-2024"}
    assert upstream.extra_query == {"trace": True, "sample": 2}


def test_build_request_without_tools_has_no_tool_choice_and_uses_empty_user_message() -> None:
    options = ProviderRuntimeOptions()
    upstream = build_request(_response_snapshot(), options=options, logger=LOGGER)
    body = build_http_body(upstream, options=options, stream=False)

    assert body == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": [{"type": "text", "text": ""}]}],
        "stream": False,
    }


def test_build_http_body_requires_normalized_parameters() -> None:
    request = AnthropicMessagesRequest(model="claude-test", messages=[], max_tokens=1)

    with pytest.raises(TypeError, match="NormalizedHostParameters"):
        build_http_body(request, options=ProviderRuntimeOptions(), stream=False)


@pytest.mark.parametrize(
    "response_request",
    [
        pytest.param(_response_snapshot(response_format={"format_type": "json_object"}), id="typed"),
        pytest.param(_response_snapshot(model_extra={"response_format": {"type": "json_object"}}), id="model-extra"),
        pytest.param(
            _response_snapshot(model_extra={"body": {"response_format": {"type": "json_object"}}}),
            id="model-nested-body",
        ),
        pytest.param(
            _response_snapshot(request_extra={"response_format": {"type": "json_object"}}), id="request-extra"
        ),
        pytest.param(
            _response_snapshot(request_extra={"body": {"response_format": {"type": "json_object"}}}),
            id="request-nested-body",
        ),
    ],
)
def test_response_format_rejection_helper_covers_all_host_entry_points(
    response_request: ResponseRequestSnapshot,
) -> None:
    with pytest.raises(ValueError, match="response_format"):
        reject_anthropic_response_format_params(response_request)


@pytest.mark.parametrize(
    "response_request",
    [
        pytest.param(_response_snapshot(response_format={"format_type": "json_object"}), id="typed"),
        pytest.param(_response_snapshot(model_extra={"response_format": {"type": "json_object"}}), id="model-extra"),
        pytest.param(
            _response_snapshot(model_extra={"body": {"response_format": {"type": "json_object"}}}),
            id="model-nested-body",
        ),
        pytest.param(
            _response_snapshot(request_extra={"response_format": {"type": "json_object"}}), id="request-extra"
        ),
        pytest.param(
            _response_snapshot(request_extra={"body": {"response_format": {"type": "json_object"}}}),
            id="request-nested-body",
        ),
    ],
)
def test_provider_request_build_rejects_unsupported_response_format(
    response_request: ResponseRequestSnapshot,
) -> None:
    with pytest.raises(ValueError, match="response_format"):
        build_request(response_request, options=ProviderRuntimeOptions(), logger=LOGGER)


@pytest.mark.asyncio
async def test_provider_rejects_response_format_without_sending_http_request() -> None:
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        del request
        request_count += 1
        return httpx.Response(200, json={})

    provider = AnthropicMessagesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError, match="response_format"):
        await provider.get_response(
            _provider_request_payload(_response_snapshot(response_format={"format_type": "json_object"}))
        )

    assert request_count == 0


def test_convert_tools_supports_nested_flat_default_and_missing_definitions() -> None:
    tool_options = [
        ToolOptionSnapshot.model_validate(_tool_definition("nested")),
        ToolOptionSnapshot.model_validate(_tool_definition("flat", nested=False)),
        ToolOptionSnapshot.model_validate({"name": "default-schema"}),
        ToolOptionSnapshot.model_validate({"name": ""}),
        ToolOptionSnapshot.model_validate({}),
    ]

    tools = convert_tools(tool_options)

    assert [tool.to_sdk_param() for tool in tools] == [
        {
            "name": "nested",
            "description": "查询信息",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
        {
            "name": "flat",
            "description": "查询信息",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
        {
            "name": "default-schema",
            "description": "",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    assert build_default_tool_choice(tools, {}) == ObjectFields(
        fields={"type": "any", "disable_parallel_tool_use": True}
    )
    assert build_default_tool_choice(tools, {"tool_choice": {"type": "auto"}}) is None
    assert build_default_tool_choice([], {}) is None


def test_convert_assistant_tool_calls_uses_id_priority_and_normalizes_arguments() -> None:
    message = MessageSnapshot.model_validate(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool-id",
                    "call_id": "call-id",
                    "function": {"name": "lookup", "arguments": '{"q":"杭州"}'},
                },
                {"id": "ignored", "function": {"name": "", "arguments": {}}},
            ],
        }
    )

    blocks = convert_assistant_tool_calls(message, options=ProviderRuntimeOptions())

    assert [block.to_sdk_param() for block in blocks] == [
        {"type": "tool_use", "id": "tool-id", "name": "lookup", "input": {"q": "杭州"}}
    ]


def test_convert_assistant_tool_calls_rejects_missing_call_id() -> None:
    message = MessageSnapshot.model_validate(
        {"role": "assistant", "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]}
    )

    with pytest.raises(ValueError, match="tool_use id"):
        convert_assistant_tool_calls(message, options=ProviderRuntimeOptions())


def test_convert_messages_associates_tool_results_and_preserves_orphans() -> None:
    messages = [
        MessageSnapshot.model_validate(
            {
                "role": "assistant",
                "parts": [{"type": "text", "text": "调用工具"}],
                "tool_calls": [{"id": "tool-1", "function": {"name": "lookup", "arguments": {"q": "杭州"}}}],
            }
        ),
        MessageSnapshot.model_validate(
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "tool_name": "lookup",
                "parts": [{"type": "text", "text": "晴"}],
            }
        ),
        MessageSnapshot.model_validate(
            {
                "role": "tool",
                "tool_call_id": "missing",
                "tool_name": "lookup",
                "parts": [{"type": "text", "text": "孤立结果"}],
            }
        ),
        MessageSnapshot.model_validate({"role": "developer", "parts": [{"type": "text", "text": "忽略"}]}),
    ]

    converted = convert_messages(messages, options=ProviderRuntimeOptions(), logger=LOGGER)

    assert [message.to_sdk_param() for message in converted] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "调用工具"},
                {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"q": "杭州"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "晴"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "工具调用结果（缺少可回放的 assistant tool_use）：lookup (missing): 孤立结果",
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    ("tool_call_id", "tool_name", "expected_label"),
    [
        pytest.param(None, None, "tool (unknown)", id="fully-missing"),
        pytest.param("  ", "  ", "tool (unknown)", id="blank"),
        pytest.param("call-1", "lookup", "lookup (call-1)", id="present"),
    ],
)
def test_orphan_tool_result_message_labels_missing_fields(
    tool_call_id: str | None,
    tool_name: str | None,
    expected_label: str,
) -> None:
    message = MessageSnapshot.model_validate(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "parts": [{"type": "text", "text": "结果"}],
        }
    )

    converted = orphan_tool_result_message(message).to_sdk_param()

    assert converted == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"工具调用结果（缺少可回放的 assistant tool_use）：{expected_label}: 结果",
            }
        ],
    }


@pytest.mark.parametrize(
    ("image_format", "expected_media_type"),
    [
        pytest.param("jpeg", "image/jpeg", id="jpeg"),
        pytest.param("png", "image/png", id="png"),
        pytest.param("gif", "image/gif", id="gif"),
        pytest.param("webp", "image/webp", id="webp"),
    ],
)
def test_anthropic_image_media_type_matrix(image_format: str, expected_media_type: str) -> None:
    assert anthropic_image_media_type(image_format) == expected_media_type


@pytest.mark.parametrize(
    ("source_format", "expected_media_type"),
    [
        pytest.param("JPEG", "image/jpeg", id="jpeg"),
        pytest.param("PNG", "image/png", id="png"),
        pytest.param("WEBP", "image/webp", id="webp"),
        pytest.param("GIF", "image/webp", id="gif-normalized-to-webp"),
    ],
)
def test_convert_image_block_accepts_supported_image_formats(
    source_format: str,
    expected_media_type: str,
) -> None:
    part = MessagePartImage(image_base64=_image_base64(source_format), image_format=source_format.lower())

    block = convert_image_block(part, options=ProviderRuntimeOptions(), logger=LOGGER)

    assert block is not None
    assert block.source.media_type == expected_media_type
    assert base64.b64decode(block.source.data, validate=True)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        pytest.param("placeholder", [{"type": "text", "text": "[图片内容不可用]"}], id="placeholder"),
        pytest.param("skip", [], id="skip"),
    ],
)
def test_convert_content_blocks_handles_invalid_user_image(
    policy: InvalidImagePolicy,
    expected: list[dict],
) -> None:
    message = MessageSnapshot.model_validate(
        {"role": "user", "parts": [{"type": "image", "image_base64": "not-base64", "image_format": "png"}]}
    )
    options = ProviderRuntimeOptions()
    options.invalid_image_policy = policy

    blocks = convert_content_blocks(message, options=options, logger=LOGGER)

    assert [block.to_sdk_param() for block in blocks] == expected


def test_convert_content_blocks_rejects_invalid_user_image_in_error_mode() -> None:
    message = MessageSnapshot.model_validate(
        {"role": "user", "parts": [{"type": "image", "image_base64": "not-base64", "image_format": "png"}]}
    )

    with pytest.raises(ValueError):
        convert_content_blocks(
            message,
            options=ProviderRuntimeOptions(invalid_image_policy="error"),
            logger=LOGGER,
        )


def test_convert_content_blocks_does_not_send_assistant_images() -> None:
    message = MessageSnapshot.model_validate(
        {"role": "assistant", "parts": [{"type": "image", "image_base64": "not-base64", "image_format": "png"}]}
    )

    assert (
        convert_content_blocks(
            message,
            options=ProviderRuntimeOptions(invalid_image_policy="error"),
            logger=LOGGER,
        )
        == []
    )


def test_convert_response_combines_all_blocks_usage_and_raw_data() -> None:
    result = convert_response(
        {
            "id": "msg-1",
            "model": "claude-test",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            },
            "content": [
                {"type": "text", "text": "A"},
                {"type": "text", "text": "B"},
                {"type": "thinking", "thinking": "想法一"},
                {"type": "thinking", "text": "想法二"},
                {"type": "redacted_thinking", "thinking": "已脱敏"},
                {"type": "unknown", "future": True},
                {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"q": "杭州"}},
            ],
        },
        options=ProviderRuntimeOptions(include_raw_data=True),
    )

    assert result.content == "AB"
    assert result.reasoning_content == "想法一\n想法二\n已脱敏"
    assert result.tool_calls[0].to_host_dict() == {
        "id": "tool-1",
        "function": {"name": "lookup", "arguments": {"q": "杭州"}},
        "extra_content": {"provider": "anthropic_messages"},
    }
    assert result.usage.to_host_dict() == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_cache_hit_tokens": 3,
        "prompt_cache_miss_tokens": 2,
    }
    assert result.raw_data is not None
    assert result.raw_data["id"] == "msg-1"
    assert result.raw_data["model"] == "claude-test"
    assert result.raw_data["stop_reason"] == "tool_use"
    assert result.raw_data["usage"] == {
        "input_tokens": "***",
        "output_tokens": "***",
        "prompt_tokens": "***",
        "completion_tokens": "***",
        "total_tokens": "***",
        "prompt_cache_hit_tokens": "***",
        "prompt_cache_miss_tokens": "***",
        "cache_read_input_tokens": "***",
        "cache_creation_input_tokens": "***",
        "input_tokens_details": "***",
        "prompt_tokens_details": "***",
    }


@pytest.mark.parametrize(
    "missing_field",
    [
        pytest.param("id", id="missing-id"),
        pytest.param("name", id="missing-name"),
        pytest.param("input", id="missing-input"),
    ],
)
def test_convert_response_rejects_malformed_tool_use(missing_field: str) -> None:
    tool_use: JsonObject = {
        "type": "tool_use",
        "id": "tool-1",
        "name": "lookup",
        "input": {},
    }
    del tool_use[missing_field]

    with pytest.raises(ValidationError):
        convert_response(
            {"content": [tool_use]},
            options=ProviderRuntimeOptions(),
        )


def test_convert_response_falls_back_to_think_tags_and_xml_tools() -> None:
    result = convert_response(
        {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<think>分析</think>正文"
                        "<tool_call><function=lookup><parameter=q>杭州</parameter></function></tool_call>"
                    ),
                }
            ]
        },
        options=ProviderRuntimeOptions(),
    )

    assert result.reasoning_content == "分析"
    assert result.content == "正文"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id.startswith("xml_tool_call_")
    assert result.tool_calls[0].function.name == "lookup"
    assert result.tool_calls[0].function.arguments == {"q": "杭州"}


def test_convert_response_reasoning_none_preserves_text_tags_and_drops_native_reasoning() -> None:
    result = convert_response(
        {
            "content": [
                {"type": "thinking", "thinking": "native"},
                {"type": "text", "text": "<think>text</think>answer"},
            ]
        },
        options=ProviderRuntimeOptions(reasoning_parse_mode="none"),
    )

    assert result.reasoning_content is None
    assert result.content == "<think>text</think>answer"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param([], id="empty"),
        pytest.param([{"type": "unknown", "future": True}], id="unknown-only"),
        pytest.param([{"type": "thinking", "thinking": "reasoning-only"}], id="reasoning-only"),
    ],
)
def test_convert_response_rejects_missing_output(content: list[JsonObject]) -> None:
    with pytest.raises(ValueError, match="Anthropic Messages"):
        convert_response({"content": content}, options=ProviderRuntimeOptions())


def _sse(events: list[tuple[str, JsonObject]]) -> bytes:
    chunks = [
        f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n" for event_name, payload in events
    ]
    return "".join(chunks).encode("utf-8")


@pytest.mark.asyncio
async def test_anthropic_stream_handles_real_event_shapes_and_compatibility_usage() -> None:
    events: list[tuple[str, JsonObject]] = [
        (
            "message_start",
            {
                "message": {
                    "id": "msg-stream",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "ignored-start-content"}],
                    "usage": {"input_tokens": 4},
                }
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "delta": {"type": "text_delta"}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "你好"}},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": -1, "delta": {"type": "thinking_delta", "thinking": "想"}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "sig"}},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '"杭州"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 99}),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": "STOP", "usage": {"output_tokens": 2}},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_delta", {"type": "message_delta", "usage": {"output_tokens": 3}}),
        ("future_event", {"type": "future_event", "future": True}),
        ("message_stop", {"type": "message_stop"}),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "text/event-stream"
        return httpx.Response(200, content=_sse(events), headers={"Content-Type": "text/event-stream"})

    provider = AnthropicMessagesProvider(
        options=ProviderRuntimeOptions(include_raw_data=True),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(
        _provider_request_payload(
            _response_snapshot(
                stream=True,
                messages=[{"role": "user", "parts": [{"type": "text", "text": "问题"}]}],
            )
        )
    )

    assert result["content"] == "你好"
    assert result["reasoning_content"] == "想"
    assert result["tool_calls"] == [
        {
            "id": "tool-1",
            "function": {"name": "lookup", "arguments": {"q": "杭州"}},
            "extra_content": {"provider": "anthropic_messages"},
        }
    ]
    assert result["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    assert result["raw_data"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_anthropic_stream_creates_placeholder_blocks_for_sparse_delta_index() -> None:
    events: list[tuple[str, JsonObject]] = [
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "第三块"}},
        ),
        ("content_block_stop", {"type": "content_block_stop"}),
        ("message_stop", {"type": "message_stop"}),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=_sse(events), headers={"Content-Type": "text/event-stream"})

    provider = AnthropicMessagesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(
        _provider_request_payload(
            _response_snapshot(
                stream=True,
                messages=[{"role": "user", "parts": [{"type": "text", "text": "问题"}]}],
            )
        )
    )

    assert result["content"] == "第三块"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name",
    [
        pytest.param("get_embedding", id="embedding"),
        pytest.param("get_audio_transcriptions", id="audio-transcription"),
    ],
)
async def test_anthropic_provider_rejects_unsupported_capabilities(method_name: str) -> None:
    provider = AnthropicMessagesProvider(options=ProviderRuntimeOptions())
    method = provider.get_embedding if method_name == "get_embedding" else provider.get_audio_transcriptions

    with pytest.raises(NotImplementedError):
        await method({})
