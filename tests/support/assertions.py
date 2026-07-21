from src.core.json_types import (
    JsonValue,
    is_json_list,
    is_json_mapping,
    mapping_to_json_object,
    normalize_json_value,
)


def assert_json_object(value: object) -> dict[str, JsonValue]:
    """断言 JSON 值为 object，并向类型检查器完成窄化。"""
    assert is_json_mapping(value)
    return mapping_to_json_object(value)


def assert_json_list(value: object) -> list[JsonValue]:
    """断言 JSON 值为 list，并向类型检查器完成窄化。"""
    assert is_json_list(value)
    return [normalize_json_value(item) for item in value]


def as_json_object(value: JsonValue) -> dict[str, JsonValue]:
    """兼容已窄化 JSON 值的 object 断言。"""
    return assert_json_object(value)


def as_json_list(value: JsonValue) -> list[JsonValue]:
    """兼容已窄化 JSON 值的 list 断言。"""
    return assert_json_list(value)
