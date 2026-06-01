from collections.abc import Iterable, Mapping
from typing import TypeAlias, cast

JsonValue: TypeAlias = str | int | float | bool | None | dict[str, object] | list[object]
JsonObject: TypeAlias = dict[str, object]
JsonArray: TypeAlias = list[object]


def empty_json_object() -> JsonObject:
    return {}


def empty_str_dict() -> dict[str, str]:
    return {}


def normalize_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return mapping_to_json_object(cast(Mapping[object, object], value))
    if isinstance(value, (list, tuple, set)):
        return [normalize_json_value(item) for item in cast(Iterable[object], value)]
    return value


def mapping_to_json_object(value: Mapping[object, object]) -> JsonObject:
    return {str(key): normalize_json_value(item) for key, item in value.items()}


def object_to_json_object(value: object) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"期望 mapping，实际为 {type(value).__name__}")
    return mapping_to_json_object(cast(Mapping[object, object], value))
