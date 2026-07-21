from src.schemas import (
    MessagePartText,
    MessagePartUnknown,
    MessageSnapshot,
    ObjectFields,
)
from tests.support.assertions import assert_json_object


def test_mapping_to_json_object_recursively_normalizes_mapping_keys() -> None:
    raw_mapping = {1: {2: "two"}, "items": [{3: "three"}]}
    normalized = assert_json_object(raw_mapping)

    assert normalized == {"1": {"2": "two"}, "items": [{"3": "three"}]}


def test_object_fields_normalizes_mapping_keys_to_strings() -> None:
    fields = ObjectFields.from_unknown({1: {2: "two"}, "two": 2})

    assert fields.to_plain_dict() == {"1": {"2": "two"}, "two": 2}


def test_message_parts_normalize_mapping_keys_to_strings() -> None:
    message = MessageSnapshot.model_validate({"parts": [{"type": "text", "text": "hello"}, {b"type": "binary"}]})

    assert isinstance(message.parts[0], MessagePartText)
    assert message.parts[0].text == "hello"
    assert isinstance(message.parts[1], MessagePartUnknown)
    assert message.parts[1].type == "unknown"
