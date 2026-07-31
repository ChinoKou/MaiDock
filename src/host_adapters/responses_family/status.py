from collections.abc import Mapping

from ...core.json_types import JsonValue, json_mapping_or_none


def incomplete_reason(response: Mapping[str, JsonValue]) -> str | None:
    """读取 Responses incomplete_details.reason。"""

    details = json_mapping_or_none(response.get("incomplete_details"))
    if details is None:
        return None
    reason = details.get("reason")
    return reason if isinstance(reason, str) else None


def is_length_incomplete(response: Mapping[str, JsonValue]) -> bool:
    """判断响应是否因达到输出 token 上限而正常截断。"""

    return response.get("status") == "incomplete" and incomplete_reason(response) == "length"
