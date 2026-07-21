from collections.abc import Iterator, Mapping

import pytest
from pydantic import BaseModel

from src.schemas import (
    FunctionCallSnapshot,
    MessagePartText,
    MessagePartUnknown,
    MessageSnapshot,
    ObjectFields,
    ResponseFormatSchemaSnapshot,
    ResponseFormatSnapshot,
    ResponseRequestSnapshot,
    ToolCallSnapshot,
    ToolOptionSnapshot,
)


class ExampleModel(BaseModel):
    answer: int


class ExampleDumpable:
    def model_dump(self, mode: str = "python") -> object:
        assert mode == "python"
        return {"source": "model-dumpable"}


class ExampleMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self._values = {"source": "mapping"}

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class SpecializedObjectFields(ObjectFields):
    pass


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(ExampleModel(answer=42), {"answer": 42}, id="pydantic-model"),
        pytest.param(ExampleDumpable(), {"source": "model-dumpable"}, id="model-dumpable"),
        pytest.param(ExampleMapping(), {"source": "mapping"}, id="mapping"),
        pytest.param(None, {}, id="none"),
    ],
)
def test_object_fields_accepts_supported_host_object_shapes(value: object, expected: dict[str, object]) -> None:
    assert ObjectFields.from_unknown(value).to_plain_dict() == expected


def test_object_fields_reuses_same_instance_and_converts_other_wrapper() -> None:
    source = ObjectFields(fields={"nested": {"value": True}})

    assert ObjectFields.from_unknown(source) is source
    converted = SpecializedObjectFields.from_unknown(source)
    assert type(converted) is SpecializedObjectFields
    assert converted.to_plain_dict() == {"nested": {"value": True}}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("not-an-object", id="string"),
        pytest.param(["not-an-object"], id="list"),
    ],
)
def test_object_fields_rejects_non_mapping_values(value: object) -> None:
    with pytest.raises(TypeError, match="(mapping|映射)"):
        ObjectFields.from_unknown(value)


def test_object_fields_validator_accepts_none_and_rejects_invalid_fields() -> None:
    assert ObjectFields.model_validate({"fields": None}).to_plain_dict() == {}

    with pytest.raises(TypeError, match="ObjectFields.fields"):
        ObjectFields.model_validate({"fields": ["invalid"]})


def test_message_snapshot_normalizes_known_unknown_and_invalid_parts() -> None:
    message = MessageSnapshot.model_validate(
        {
            "role": "user",
            "parts": [
                {"type": "text", "text": "hello", "future": "ignored"},
                {"type": "future", "payload": {"ignored": True}},
                "not-a-part",
            ],
            "future_message_field": "ignored",
        }
    )

    assert len(message.parts) == 2
    assert isinstance(message.parts[0], MessagePartText)
    assert message.parts[0].text == "hello"
    assert isinstance(message.parts[1], MessagePartUnknown)
    assert message.parts[1].type == "future"
    assert "future_message_field" not in message.model_fields_set


@pytest.mark.parametrize(
    "field",
    [
        pytest.param("parts", id="parts"),
        pytest.param("tool_calls", id="tool-calls"),
    ],
)
def test_message_snapshot_treats_non_list_collection_as_empty(field: str) -> None:
    message = MessageSnapshot.model_validate({field: {"not": "a-list"}})

    assert getattr(message, field) == []


@pytest.mark.parametrize(
    ("tool_call", "expected"),
    [
        pytest.param({"id": " primary ", "call_id": "legacy"}, "primary", id="id-first"),
        pytest.param({"call_id": " legacy "}, "legacy", id="call-id-fallback"),
        pytest.param({}, "", id="missing"),
    ],
)
def test_tool_call_id_resolution_priority(tool_call: dict[str, str], expected: str) -> None:
    assert ToolCallSnapshot.model_validate(tool_call).resolved_call_id() == expected


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param({"city": "上海"}, {"city": "上海"}, id="object"),
        pytest.param('{"city":"上海"}', '{"city":"上海"}', id="string"),
        pytest.param(None, None, id="none"),
    ],
)
def test_function_call_accepts_all_core_argument_shapes(arguments: object, expected: object) -> None:
    function_call = FunctionCallSnapshot.model_validate({"arguments": arguments})

    if isinstance(function_call.arguments, ObjectFields):
        assert function_call.arguments.to_plain_dict() == expected
    else:
        assert function_call.arguments == expected


def test_nested_and_flat_tool_definitions_produce_same_function_contract() -> None:
    nested = ToolOptionSnapshot.model_validate(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "nested",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    )
    flat = ToolOptionSnapshot.model_validate(
        {
            "name": "lookup",
            "description": "flat",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    )

    assert nested.function_definition().name == flat.function_definition().name == "lookup"
    assert nested.function_definition().description == "nested"
    assert flat.function_definition().description == "flat"
    assert (
        nested.function_definition().parameters.to_plain_dict() == flat.function_definition().parameters.to_plain_dict()
    )


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        pytest.param(
            {
                "format_type": "json_schema",
                "schema": {
                    "name": "result",
                    "description": "contract",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            ResponseFormatSchemaSnapshot,
            id="named-schema",
        ),
        pytest.param(
            {"format_type": "json_schema", "schema": {"type": "object", "properties": {}}},
            ObjectFields,
            id="raw-schema",
        ),
    ],
)
def test_response_format_distinguishes_named_and_raw_schema(
    payload: dict[str, object],
    expected_type: type[ResponseFormatSchemaSnapshot | ObjectFields],
) -> None:
    response_format = ResponseFormatSnapshot.model_validate(payload)

    assert isinstance(response_format.schema_, expected_type)


def test_response_format_accepts_existing_schema_models_and_none() -> None:
    named = ResponseFormatSchemaSnapshot.model_validate({"name": "result", "schema": {"type": "object"}})
    wrapped = ResponseFormatSnapshot(schema=named)
    raw = ObjectFields(fields={"type": "object"})

    assert wrapped.schema_ is named
    assert ResponseFormatSnapshot(schema=raw).schema_ is raw
    assert ResponseFormatSnapshot(schema=None).schema_ is None


def test_response_request_defaults_and_invalid_collection_compatibility() -> None:
    default_request = ResponseRequestSnapshot()
    invalid_request = ResponseRequestSnapshot.model_validate(
        {"message_list": {"not": "a-list"}, "tool_options": "not-a-list"}
    )

    assert default_request.message_list == []
    assert default_request.tool_options == []
    assert invalid_request.message_list == []
    assert invalid_request.tool_options == []


def test_empty_tool_parameters_are_normalized_to_default_schema() -> None:
    option = ToolOptionSnapshot.model_validate({"name": "lookup", "parameters": {}})

    assert option.parameters.to_plain_dict() == {"type": "object", "properties": {}}
