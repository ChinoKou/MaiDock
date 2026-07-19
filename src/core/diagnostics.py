import re
from collections.abc import Mapping

from ..i18n import runtime_expected, runtime_subject, translate
from .json_types import (
    JsonValue,
    is_json_iterable,
    is_json_mapping,
    json_mapping_or_none,
    mapping_to_json_object,
    normalize_json_value,
)

SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "token", "secret", "password")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?P<prefix>\b(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|token|secret|password)"
    r"\b[\x22\x27]?\s*[:=]\s*)(?:Bearer\s+)?[\x22\x27]?[^,\s}\x22\x27]+"
)
_DATA_URI_PATTERN = re.compile(r"(?i)^data:[^,;]+(?:;[^,]*)?;base64,")
_BARE_BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{128,}={0,2}")


def _sanitize_bytes(value: bytes | bytearray, *, max_text_length: int) -> str:
    del max_text_length
    return f"<bytes:{len(value)}>"


def _sanitize_text(value: str, *, max_text_length: int, detect_base64: bool = True) -> str:
    redacted = _BEARER_PATTERN.sub("Bearer ***", value)
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(r"\g<prefix>***", redacted)
    if _DATA_URI_PATTERN.match(redacted):
        prefix, _, payload = redacted.partition(",")
        return f"{prefix},<base64:{len(payload)}>"
    if detect_base64 and _BARE_BASE64_PATTERN.fullmatch(redacted):
        return f"<base64:{len(redacted)}>"
    if len(redacted) > max_text_length:
        return f"{redacted[:max_text_length]}...<truncated:{len(redacted)}>"
    return redacted


def sanitize_for_log(value: object, *, max_text_length: int = 300):
    """递归脱敏用于日志或 raw_data 的对象。"""

    if isinstance(value, (bytes, bytearray)):
        return _sanitize_bytes(value, max_text_length=max_text_length)
    if isinstance(value, memoryview):
        return f"<bytes:{value.nbytes}>"
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": _sanitize_text(str(value), max_text_length=max_text_length, detect_base64=False),
        }
    if is_json_mapping(value):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = str(key)
            lowered_key = normalized_key.lower()
            if any(secret_key in lowered_key for secret_key in SECRET_KEY_PARTS):
                sanitized[normalized_key] = "***"
                continue
            if isinstance(item, str) and any(
                text_key in lowered_key for text_key in ("prompt", "instruction", "content", "text")
            ):
                sanitized[normalized_key] = _sanitize_text(
                    item,
                    max_text_length=max_text_length,
                    detect_base64=False,
                )
                continue
            sanitized[normalized_key] = sanitize_for_log(item, max_text_length=max_text_length)
        return sanitized
    if is_json_iterable(value):
        return [sanitize_for_log(item, max_text_length=max_text_length) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value, max_text_length=max_text_length)
    return value


def sanitize_upstream_detail(value: object, *, max_text_length: int = 300) -> str:
    """把任意上游详情转换为可安全透传给 Host 的文本。"""

    return str(sanitize_for_log(value, max_text_length=max_text_length))


def sanitize_json_value(value: object, *, max_text_length: int = 300) -> object:
    return normalize_json_value(sanitize_for_log(value, max_text_length=max_text_length))


def sanitize_json_object(value: Mapping[str, JsonValue] | dict, *, max_text_length: int = 300) -> dict:
    sanitized = sanitize_for_log(value, max_text_length=max_text_length)
    sanitized_mapping = json_mapping_or_none(sanitized)
    if sanitized_mapping is None:
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject=runtime_subject("sanitized_value"),
                expected=runtime_expected("mapping"),
                actual=type(sanitized).__name__,
            )
        )
    return mapping_to_json_object(sanitized_mapping)


def extract_error_body(error: BaseException) -> object | None:
    """尽量从上游异常中提取响应体。"""

    candidates = [error, getattr(error, "__cause__", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        response = getattr(candidate, "response", None)
        if response is not None:
            response_json = getattr(response, "json", None)
            if callable(response_json):
                try:
                    return sanitize_for_log(response_json())
                except Exception:
                    pass
            response_text = getattr(response, "text", None)
            if response_text not in (None, ""):
                return sanitize_for_log(str(response_text))
            response_content = getattr(response, "content", None)
            if response_content not in (None, b"", ""):
                return sanitize_for_log(response_content)
        body = getattr(candidate, "body", None)
        if body not in (None, "", b""):
            return sanitize_for_log(body)
    return None


def build_status_error_message(provider_label: str, error: BaseException) -> str:
    """构造适合透传到 Host 的上游状态错误。"""

    status_code = getattr(error, "status_code", None)
    message_parts: list[str] = []
    message = getattr(error, "message", None)
    if message:
        message_parts.append(sanitize_upstream_detail(message))
    error_body = extract_error_body(error)
    if error_body not in (None, ""):
        message_parts.append(sanitize_upstream_detail(error_body))
    if message_parts:
        return translate(
            "runtime.error.upstream_status",
            provider=provider_label,
            details=" | ".join(message_parts),
        )
    if status_code is not None:
        return translate(
            "runtime.error.upstream_status",
            provider=provider_label,
            details=f"HTTP {status_code}",
        )
    return translate(
        "runtime.error.upstream_call",
        provider=provider_label,
        details=sanitize_upstream_detail(error),
    )


def build_connection_error_message(provider_label: str, error: BaseException) -> str:
    """构造连接错误信息。"""

    return translate(
        "runtime.error.upstream_connect",
        provider=provider_label,
        details=sanitize_upstream_detail(error),
    )


def build_parse_error_message(provider_label: str, message: str) -> str:
    """构造解析错误信息。"""

    return translate("runtime.error.parse", provider=provider_label, message=message)


def compact_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """复制非 None 字段，便于生成 raw_data 摘要。"""

    return {str(key): item for key, item in value.items() if item is not None}
