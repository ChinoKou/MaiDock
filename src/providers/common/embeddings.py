import base64
import math
import struct

from ...core.diagnostics import build_parse_error_message
from ...core.json_types import json_list_or_none
from ...i18n import runtime_expected, runtime_item, runtime_subject, translate
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
                message = translate(
                    "runtime.error.expected_type",
                    subject=f"embedding[{index}]",
                    expected=runtime_expected("finite_number"),
                    actual=item,
                )
                raise HttpxProviderParseError(build_parse_error_message(provider_label, message))
        return embedding

    raw_embedding = json_list_or_none(value)
    if raw_embedding is None:
        message = translate(
            "runtime.error.required",
            subject=runtime_subject("response"),
            field=runtime_item("embedding_array"),
        )
        raise HttpxProviderParseError(build_parse_error_message(provider_label, message))
    embedding: list[float] = []
    for index, item in enumerate(raw_embedding):
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            message = translate(
                "runtime.error.expected_type",
                subject=f"embedding[{index}]",
                expected=runtime_expected("float_compatible_value"),
                actual=type(item).__name__,
            )
            raise HttpxProviderParseError(build_parse_error_message(provider_label, message))
        try:
            value = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            message = translate(
                "runtime.error.expected_type",
                subject=f"embedding[{index}]",
                expected=runtime_expected("float_compatible_value"),
                actual=type(item).__name__,
            )
            raise HttpxProviderParseError(build_parse_error_message(provider_label, message)) from exc
        if not math.isfinite(value):
            message = translate(
                "runtime.error.expected_type",
                subject=f"embedding[{index}]",
                expected=runtime_expected("finite_number"),
                actual=value,
            )
            raise HttpxProviderParseError(build_parse_error_message(provider_label, message))
        embedding.append(value)
    return embedding
