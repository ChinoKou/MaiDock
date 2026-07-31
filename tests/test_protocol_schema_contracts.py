import pytest
from pydantic import ValidationError

from src.schemas import (
    AnthropicImageBlock,
    AnthropicImageMediaType,
    AnthropicImageSource,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicRawData,
    AnthropicResponseContentBlock,
    AnthropicResponseSnapshot,
    AnthropicTextBlock,
    AnthropicTool,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    GenericUsageSnapshot,
    ObjectFields,
    OpenAIEasyInputMessage,
    OpenAIFunctionCallInputItem,
    OpenAIFunctionCallOutputItem,
    OpenAIInputImageBlock,
    OpenAIInputMessage,
    OpenAIInputTextBlock,
    OpenAIOutputTextBlock,
    OpenAIRawData,
    OpenAIResponseOutputContentBlock,
    OpenAIResponseOutputMessageItem,
    OpenAIResponseSnapshot,
    OpenAIResponsesRequest,
    OpenAIResponsesTool,
    OpenAITextConfig,
    OpenAITextFormatConfig,
)
from tests.support.assertions import json_int_at


def test_openai_input_items_and_tools_match_responses_wire_shapes() -> None:
    user = OpenAIInputMessage(
        role="user",
        content=[
            OpenAIInputTextBlock(text="hello"),
            OpenAIInputImageBlock(image_url="data:image/png;base64,aW1hZ2U=", detail="high"),
        ],
    )
    assistant = OpenAIEasyInputMessage(content="previous")
    replay = OpenAIResponseOutputMessageItem(id="msg_1", content=[OpenAIOutputTextBlock(text="answer")])
    call = OpenAIFunctionCallInputItem(
        call_id="call_1",
        name="lookup",
        arguments='{"q":"weather"}',
        id="item_1",
        status="completed",
    )
    output = OpenAIFunctionCallOutputItem(call_id="call_1", output="sunny")
    tool = OpenAIResponsesTool(
        name="lookup",
        description="Lookup data",
        parameters=ObjectFields(fields={"type": "object", "properties": {}}),
        strict=True,
    )
    request = OpenAIResponsesRequest(
        model="contract-model",
        input=[user, assistant, replay, call, output],
        tools=[tool],
    )

    assert request.input_params() == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U=", "detail": "high"},
            ],
        },
        {"role": "assistant", "content": "previous"},
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"weather"}',
            "id": "item_1",
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    ]
    assert request.tool_params() == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup data",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        }
    ]


def test_openai_function_call_omits_optional_wire_fields() -> None:
    call = OpenAIFunctionCallInputItem(call_id="call_1", name="lookup", arguments="{}")

    assert call.to_sdk_param() == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
    }


@pytest.mark.parametrize(
    ("format_config", "expected"),
    [
        pytest.param(OpenAITextFormatConfig(type="text"), {"type": "text"}, id="text"),
        pytest.param(OpenAITextFormatConfig(type="json_object"), {"type": "json_object"}, id="json-object"),
        pytest.param(
            OpenAITextFormatConfig(
                type="json_schema",
                name=" result ",
                description="contract",
                schema=ObjectFields(fields={"type": "object"}),
                strict=True,
            ),
            {
                "type": "json_schema",
                "name": "result",
                "description": "contract",
                "schema": {"type": "object"},
                "strict": True,
            },
            id="json-schema",
        ),
        pytest.param(
            OpenAITextFormatConfig(
                type="json_schema",
                name="minimal",
                schema=ObjectFields(fields={"type": "object"}),
            ),
            {"type": "json_schema", "name": "minimal", "schema": {"type": "object"}},
            id="minimal-json-schema",
        ),
    ],
)
def test_openai_text_format_wire_contract(
    format_config: OpenAITextFormatConfig,
    expected: dict[str, object],
) -> None:
    assert format_config.to_sdk_param() == expected


@pytest.mark.parametrize(
    "format_config",
    [
        pytest.param(
            OpenAITextFormatConfig(type="json_schema", name=" ", schema=ObjectFields(fields={})),
            id="blank-name",
        ),
        pytest.param(OpenAITextFormatConfig(type="json_schema", name="result"), id="missing-schema"),
    ],
)
def test_openai_json_schema_format_requires_name_and_schema(format_config: OpenAITextFormatConfig) -> None:
    with pytest.raises(ValueError, match="(name|schema)"):
        format_config.to_sdk_param()


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(OpenAITextConfig(), {}, id="empty"),
        pytest.param(OpenAITextConfig(verbosity="high"), {"verbosity": "high"}, id="verbosity"),
        pytest.param(
            OpenAITextConfig(format=OpenAITextFormatConfig(type="json_object"), verbosity="low"),
            {"format": {"type": "json_object"}, "verbosity": "low"},
            id="format-and-verbosity",
        ),
    ],
)
def test_openai_text_config_includes_only_configured_fields(
    config: OpenAITextConfig,
    expected: dict[str, object],
) -> None:
    assert config.to_sdk_param() == expected


def test_openai_response_snapshot_tolerates_unknown_fields_and_invalid_lists() -> None:
    response = OpenAIResponseSnapshot.model_validate(
        {
            "id": "resp_1",
            "model": "contract-model",
            "status": "completed",
            "output_text": "answer",
            "output": [
                {
                    "type": "message",
                    "content": {"not": "a-list"},
                    "summary": "not-a-list",
                    "future_item_field": True,
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "input_tokens_details": {"cached_tokens": 3},
            },
            "future_response_field": True,
        }
    )

    assert len(response.output) == 1
    assert response.output[0].content == []
    assert response.output[0].summary == []
    assert "future_item_field" not in response.output[0].model_fields_set
    assert response.usage.input_tokens_details.to_plain_dict() == {"cached_tokens": 3}
    assert "future_response_field" not in response.model_fields_set


def test_openai_response_output_item_default_content_lists_are_isolated() -> None:
    response = OpenAIResponseSnapshot.model_validate({"output": [{"type": "message"}, {"type": "reasoning"}]})
    response.output[0].content.append(OpenAIResponseOutputContentBlock(type="output_text", text="first"))

    assert response.output[0].summary == []
    assert response.output[1].content == []
    assert response.output[1].summary == []


def test_openai_raw_data_serializes_generic_usage() -> None:
    raw_data = OpenAIRawData(
        id="resp_1",
        model="contract-model",
        status="completed",
        usage=GenericUsageSnapshot(input_tokens=3, output_tokens=2, total_tokens=5),
    )

    assert raw_data.to_host_dict()["usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 5,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "input_tokens_details": {"fields": {}},
        "prompt_tokens_details": {"fields": {}},
    }


def test_openai_request_and_tool_default_factories_match_protocol_defaults() -> None:
    request = OpenAIResponsesRequest(model="contract-model", input=[])
    tool = OpenAIResponsesTool(name="lookup")
    tool_without_strict = OpenAIResponsesTool(name="lookup", strict=None)
    response = OpenAIResponseSnapshot()

    assert request.tools == []
    assert tool.parameters.to_plain_dict() == {"type": "object", "properties": {}}
    assert tool.to_sdk_param()["strict"] is False
    assert "strict" not in tool_without_strict.to_sdk_param()
    assert response.output == []


@pytest.mark.parametrize(
    "media_type",
    [
        pytest.param("image/jpeg", id="jpeg"),
        pytest.param("image/png", id="png"),
        pytest.param("image/gif", id="gif"),
        pytest.param("image/webp", id="webp"),
    ],
)
def test_anthropic_image_media_types_match_messages_protocol(media_type: AnthropicImageMediaType) -> None:
    source = AnthropicImageSource(media_type=media_type, data="aW1hZ2U=")

    assert AnthropicImageBlock(source=source).to_sdk_param() == {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": "aW1hZ2U="},
    }


def test_anthropic_messages_and_tools_match_wire_shapes() -> None:
    user = AnthropicMessage(
        role="user",
        content=[
            AnthropicTextBlock(text="hello"),
            AnthropicToolResultBlock(tool_use_id="tool_1", content="sunny"),
        ],
    )
    assistant = AnthropicMessage(
        role="assistant",
        content=[AnthropicToolUseBlock(id="tool_1", name="lookup", input={"city": "上海"})],
    )
    tool = AnthropicTool(
        name="lookup",
        description="Lookup data",
        input_schema=ObjectFields(fields={"type": "object", "properties": {}}),
    )
    request = AnthropicMessagesRequest(
        model="claude-contract",
        messages=[user, assistant],
        max_tokens=256,
        system="contract system",
        temperature=0.2,
        tools=[tool],
        tool_choice=ObjectFields(fields={"type": "auto", "disable_parallel_tool_use": True}),
    )

    assert request.message_params() == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "sunny"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"city": "上海"}}],
        },
    ]
    assert request.tool_params() == [
        {
            "name": "lookup",
            "description": "Lookup data",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    assert request.tool_choice is not None
    assert request.tool_choice.to_plain_dict() == {"type": "auto", "disable_parallel_tool_use": True}


def test_anthropic_response_and_raw_data_preserve_usage_and_unknown_blocks() -> None:
    response = AnthropicResponseSnapshot.model_validate(
        {
            "id": "msg_1",
            "model": "claude-contract",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 6,
                "cache_read_input_tokens": 8,
                "cache_creation_input_tokens": 4,
            },
            "content": [
                {"type": "text", "text": "answer", "future": "ignored"},
                {"type": "future_block", "future": True},
            ],
        }
    )
    raw_data = AnthropicRawData(
        id=response.id,
        model=response.model,
        stop_reason=response.stop_reason,
        usage=response.usage,
    )

    assert [block.type for block in response.content] == ["text", "future_block"]
    assert response.usage.cache_read_input_tokens == 8
    assert response.usage.cache_creation_input_tokens == 4
    assert json_int_at(raw_data.to_host_dict(), "usage", "cache_read_input_tokens") == 8


def test_anthropic_response_treats_non_list_content_as_empty() -> None:
    response = AnthropicResponseSnapshot.model_validate({"content": {"not": "a-list"}})
    default_response = AnthropicResponseSnapshot()
    default_request = AnthropicMessagesRequest(model="claude-contract", messages=[], max_tokens=64)

    assert response.content == []
    assert default_response.content == []
    assert default_request.tools == []


@pytest.mark.parametrize(
    "missing_field",
    [
        pytest.param("id", id="missing-id"),
        pytest.param("name", id="missing-name"),
        pytest.param("input", id="missing-input"),
    ],
)
def test_anthropic_response_tool_use_requires_protocol_fields(missing_field: str) -> None:
    payload: dict[str, object] = {
        "type": "tool_use",
        "id": "tool_1",
        "name": "lookup",
        "input": {"city": "上海"},
    }
    del payload[missing_field]

    with pytest.raises(ValidationError):
        AnthropicResponseContentBlock.model_validate(payload)


@pytest.mark.parametrize(
    "null_field",
    [
        pytest.param("id", id="null-id"),
        pytest.param("name", id="null-name"),
        pytest.param("input", id="null-input"),
    ],
)
def test_anthropic_response_tool_use_rejects_null_protocol_fields(null_field: str) -> None:
    payload: dict[str, object] = {
        "type": "tool_use",
        "id": "tool_1",
        "name": "lookup",
        "input": {"city": "上海"},
    }
    payload[null_field] = None

    with pytest.raises(ValidationError):
        AnthropicResponseContentBlock.model_validate(payload)


@pytest.mark.parametrize("invalid_input", [pytest.param([], id="list"), pytest.param("{}", id="string")])
def test_anthropic_response_tool_use_rejects_non_object_input(invalid_input: object) -> None:
    with pytest.raises(ValidationError):
        AnthropicResponseContentBlock.model_validate(
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "lookup",
                "input": invalid_input,
            }
        )


def test_anthropic_response_tool_use_allows_empty_input_without_affecting_unknown_blocks() -> None:
    tool_use = AnthropicResponseContentBlock.model_validate(
        {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {}}
    )
    unknown = AnthropicResponseContentBlock.model_validate({"type": "future_block", "future": True})

    assert tool_use.input == {}
    assert unknown.type == "future_block"
