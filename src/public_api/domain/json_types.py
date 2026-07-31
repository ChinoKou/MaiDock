from collections.abc import Mapping
from math import isfinite
from typing import TypeGuard

type PublicJsonValue = str | int | float | bool | None | dict[str, "PublicJsonValue"] | list["PublicJsonValue"]
type PublicJsonObject = dict[str, PublicJsonValue]
type PublicRpcValue = PublicJsonValue | bytes | dict[str, "PublicRpcValue"] | list["PublicRpcValue"]
type PublicRpcObject = dict[str, PublicRpcValue]


def is_public_json_object(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def normalize_public_json(value: object) -> PublicJsonValue:
    """严格窄化有限 JSON，拒绝隐式字符串化。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON 数值必须是有限值")
        return value
    if is_public_json_object(value):
        return {key: normalize_public_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_public_json(item) for item in value]
    raise TypeError(f"值不是 JSON 类型: {type(value).__name__}")


def normalize_public_json_object(value: object) -> PublicJsonObject:
    if not is_public_json_object(value):
        raise TypeError("值必须是 JSON object")
    return {key: normalize_public_json(item) for key, item in value.items()}


def normalize_public_rpc(value: object) -> PublicRpcValue:
    """窄化 MsgPack RPC 值；仅比 JSON 多允许 bytes。"""

    if isinstance(value, bytes):
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("RPC 数值必须是有限值")
        return value
    if is_public_json_object(value):
        return {key: normalize_public_rpc(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [normalize_public_rpc(item) for item in value]
    raise TypeError(f"值不是可序列化 RPC 类型: {type(value).__name__}")


def normalize_public_rpc_object(value: object) -> PublicRpcObject:
    if not is_public_json_object(value):
        raise TypeError("RPC 响应必须是 object")
    return {key: normalize_public_rpc(item) for key, item in value.items()}
