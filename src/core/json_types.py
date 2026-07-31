from collections.abc import Iterable, Mapping
from typing import TypeGuard

from ..i18n import runtime_expected, runtime_subject, translate

type JsonValue = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]


def is_json_mapping(value: object) -> TypeGuard[Mapping[str, JsonValue]]:
    return isinstance(value, Mapping)


def is_json_iterable(value: object) -> TypeGuard[Iterable[JsonValue]]:
    return isinstance(value, (list, tuple))


def is_json_list(value: object) -> TypeGuard[list[JsonValue]]:
    return isinstance(value, list)


def json_mapping_or_none(value: object) -> Mapping[str, JsonValue] | None:
    if is_json_mapping(value):
        return value
    return None


def json_list_or_none(value: object) -> list[JsonValue] | None:
    if is_json_list(value):
        return value
    return None


def json_array(items: Iterable[JsonValue]) -> list[JsonValue]:
    """把一串 JSON 值收成 list[JsonValue]，用于把精确类型的列表放进 JSON 槽位。

    list 在类型系统里是不变的：`list[dict[str, JsonValue]]` 并不是 `list[JsonValue]`，
    所以一个"元素类型更精确"的列表反而不能直接赋给 JsonValue。形参用协变的 Iterable
    接住这类列表，返回时收成不变的 list[JsonValue]。

    这是纯类型层面的收拢，运行时只是浅拷贝一次；调用点因此都是可 grep 的显式动作，
    而不是各处零散地重新标注中间变量。
    """

    return list(items)


def mapping_field(value: Mapping[str, JsonValue], key: str) -> Mapping[str, JsonValue] | None:
    return json_mapping_or_none(value.get(key))


def list_field(value: Mapping[str, JsonValue], key: str) -> list[JsonValue] | None:
    return json_list_or_none(value.get(key))


def string_field(value: Mapping[str, JsonValue], key: str) -> str | None:
    field_value = value.get(key)
    return field_value if isinstance(field_value, str) else None


def normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_json_mapping(value):
        return mapping_to_json_object(value)
    if is_json_iterable(value):
        return [normalize_json_value(item) for item in value]
    return str(value)


def mapping_to_json_object(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {str(key): normalize_json_value(item) for key, item in value.items()}


def value_to_json_object(value: object) -> dict[str, JsonValue]:
    if not is_json_mapping(value):
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject=runtime_subject("json_value"),
                expected=runtime_expected("mapping"),
                actual=type(value).__name__,
            )
        )
    return mapping_to_json_object(value)
