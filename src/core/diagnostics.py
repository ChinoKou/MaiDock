from collections.abc import Iterable, Mapping
from typing import cast

SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "token", "secret", "password")
IMAGE_DATA_PREFIX = "data:image/"


def _sanitize_bytes(value: bytes | bytearray, *, max_text_length: int) -> str:
    del max_text_length
    return f"<bytes:{len(value)}>"


def sanitize_for_log(value: object, *, max_text_length: int = 300) -> object:
    """递归脱敏用于日志或 raw_data 的对象。"""

    if isinstance(value, (bytes, bytearray)):
        return _sanitize_bytes(value, max_text_length=max_text_length)
    if isinstance(value, memoryview):
        return f"<bytes:{value.nbytes}>"
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            normalized_key = str(key)
            lowered_key = normalized_key.lower()
            if any(secret_key in lowered_key for secret_key in SECRET_KEY_PARTS):
                sanitized[normalized_key] = "***"
                continue
            sanitized[normalized_key] = sanitize_for_log(item, max_text_length=max_text_length)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        iterable = cast(Iterable[object], value)
        return [sanitize_for_log(item, max_text_length=max_text_length) for item in iterable]
    if isinstance(value, str):
        if value.startswith(IMAGE_DATA_PREFIX):
            return f"{value[:48]}...<base64:{len(value)}>"
        if len(value) > max_text_length:
            return f"{value[:max_text_length]}...<truncated:{len(value)}>"
    return value


def extract_error_body(error: BaseException) -> object | None:
    """尽量从 SDK 异常中提取上游响应体。"""

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
        message_parts.append(str(message))
    error_body = extract_error_body(error)
    if error_body not in (None, ""):
        message_parts.append(str(error_body))
    if message_parts:
        return f"{provider_label} 上游接口返回错误: " + " | ".join(message_parts)
    if status_code is not None:
        return f"{provider_label} 上游接口返回状态码 {status_code}"
    return f"{provider_label} 上游接口调用失败: {error}"


def build_connection_error_message(provider_label: str, error: BaseException) -> str:
    """构造连接错误信息。"""

    return f"{provider_label} 上游接口连接失败: {error}"


def build_parse_error_message(provider_label: str, message: str) -> str:
    """构造解析错误信息。"""

    return f"{provider_label} 响应解析失败: {message}"


def compact_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """复制非 None 字段，便于生成 raw_data 摘要。"""

    return {str(key): item for key, item in value.items() if item is not None}
