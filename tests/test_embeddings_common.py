import base64
import math
import struct

from src.host_adapters.common.embeddings import (
    coerce_embedding_vector,
    decode_base64_embedding,
)


def _encode_floats(floats: list[float]) -> str:
    """Encode a list of floats as a base64 string. Each float is little-endian float32."""
    raw = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(raw).decode()


def test_decode_base64_embedding_known_vector():
    """decode_base64_embedding 将 base64 编码的 float32 数组解码为 list[float]"""
    floats = [0.1, 0.2, 0.3, 0.5]
    encoded = _encode_floats(floats)

    result = decode_base64_embedding(encoded)

    assert len(result) == 4
    for actual, expected in zip(result, floats, strict=True):
        assert math.isclose(actual, expected, rel_tol=1e-6)


def test_decode_base64_embedding_single_float():
    """单 float 向量"""
    encoded = _encode_floats([42.0])
    result = decode_base64_embedding(encoded)
    assert len(result) == 1
    assert math.isclose(result[0], 42.0, rel_tol=1e-6)


def test_decode_base64_embedding_empty_returns_empty():
    """空 base64 字符串返回空列表"""
    result = decode_base64_embedding("")
    assert result == []


def test_coerce_embedding_vector_base64_string_decoded():
    """encoding_format='base64' 且 value 为 str 时自动 decode"""
    floats = [0.1, 0.2, 0.3, 0.5]
    encoded = _encode_floats(floats)

    result = coerce_embedding_vector(encoded, provider_label="test", encoding_format="base64")

    assert len(result) == 4
    for actual, expected in zip(result, floats, strict=True):
        assert math.isclose(actual, expected, rel_tol=1e-6)


def test_coerce_embedding_vector_float_array_backward_compat():
    """不传 encoding_format 时走原有 float 数组路径"""
    result = coerce_embedding_vector([0.1, 0.2], provider_label="test")
    assert result == [0.1, 0.2]


def test_coerce_embedding_vector_mixed_types():
    """encoding_format 未指定时，str 类型的浮点数也能正常转换"""
    result = coerce_embedding_vector(["0.1", 0.2, "3"], provider_label="test")
    assert result == [0.1, 0.2, 3.0]


def test_coerce_embedding_vector_rejects_non_finite_in_base64():
    """base64 decode 后含非有限数值时报错"""
    # NaN: exponent all 1s, mantissa non-zero → 0x7fc00000 (little-endian)
    nan_bytes = struct.pack("<I", 0x7FC00000)
    encoded = base64.b64encode(nan_bytes).decode()

    import pytest

    from src.host_adapters.common.httpx import HttpxProviderParseError

    with pytest.raises(HttpxProviderParseError, match=r"embedding\[0\].*nan"):
        coerce_embedding_vector(encoded, provider_label="test", encoding_format="base64")
