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


def _navigate(value: JsonValue, path: tuple[str | int, ...]) -> JsonValue:
    """沿路径逐跳下钻，用 isinstance 窄化而不复制容器。

    上面几个 assert_* 会把 Mapping/Sequence 重建成纯 JSON 容器，返回的是副本；
    这里必须原样返回被下钻到的那个对象，调用方才能对它做 del/赋值并让改动落回原结构。
    """

    current = value
    for depth, key in enumerate(path):
        walked = path[:depth]
        if isinstance(key, str):
            assert isinstance(current, dict), f"路径 {walked} 处不是 object：{type(current).__name__}"
            assert key in current, f"路径 {walked} 的 object 缺少键 {key!r}"
        else:
            assert isinstance(current, list), f"路径 {walked} 处不是 list：{type(current).__name__}"
            assert -len(current) <= key < len(current), f"路径 {walked} 的 list 下标 {key} 越界"
        current = current[key]
    return current


def json_object_at(value: JsonValue, *path: str | int) -> dict[str, JsonValue]:
    """下钻到路径处的 object 本体（非副本），可直接增删改。"""
    resolved = _navigate(value, path)
    assert isinstance(resolved, dict), f"路径 {path} 处不是 object：{type(resolved).__name__}"
    return resolved


def json_list_at(value: JsonValue, *path: str | int) -> list[JsonValue]:
    """下钻到路径处的 list 本体（非副本），可直接增删改。"""
    resolved = _navigate(value, path)
    assert isinstance(resolved, list), f"路径 {path} 处不是 list：{type(resolved).__name__}"
    return resolved


def json_str_at(value: JsonValue, *path: str | int) -> str:
    """下钻并断言路径处是字符串。"""
    resolved = _navigate(value, path)
    assert isinstance(resolved, str), f"路径 {path} 处不是 str：{type(resolved).__name__}"
    return resolved


def json_int_at(value: JsonValue, *path: str | int) -> int:
    """下钻并断言路径处是整数；bool 是 int 子类，这里显式排除。"""
    resolved = _navigate(value, path)
    assert isinstance(resolved, int) and not isinstance(resolved, bool), (
        f"路径 {path} 处不是 int：{type(resolved).__name__}"
    )
    return resolved
