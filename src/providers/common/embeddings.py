import base64
import math
import struct

from ...core.diagnostics import build_parse_error_message
from ...core.json_types import json_list_or_none
from .httpx import HttpxProviderParseError


def decode_base64_embedding(encoded: str) -> list[float]:
    """将 base64 编码的 embedding 字符串按小端 float32 解码为浮点向量。"""
    raw_bytes = base64.b64decode(encoded)
    count = len(raw_bytes) // 4
    return list(struct.unpack(f"<{count}f", raw_bytes))


def coerce_embedding_vector(value: object, *, provider_label: str, encoding_format: str | None = None) -> list[float]:
    """校验并将 embedding 候选值强制转换为有限浮点向量，base64 编码字符串会自动解码。"""
    if encoding_format == "base64" and isinstance(value, str):
        embedding = decode_base64_embedding(value)
        for index, item in enumerate(embedding):
            if not math.isfinite(item):
                raise HttpxProviderParseError(
                    build_parse_error_message(provider_label, f"embedding[{index}] 不是有限数值")
                )
        return embedding

    raw_embedding = json_list_or_none(value)
    if raw_embedding is None:
        raise HttpxProviderParseError(build_parse_error_message(provider_label, "缺少 embedding 数组"))
    embedding: list[float] = []
    for index, item in enumerate(raw_embedding):
        if not isinstance(item, (str, int, float)):
            raise HttpxProviderParseError(
                build_parse_error_message(
                    provider_label,
                    f"embedding[{index}] 无法转换为 float，类型为 {type(item).__name__}",
                )
            )
        try:
            value = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HttpxProviderParseError(
                build_parse_error_message(
                    provider_label,
                    f"embedding[{index}] 无法转换为 float，类型为 {type(item).__name__}",
                )
            ) from exc
        if not math.isfinite(value):
            raise HttpxProviderParseError(build_parse_error_message(provider_label, f"embedding[{index}] 不是有限数值"))
        embedding.append(value)
    return embedding
