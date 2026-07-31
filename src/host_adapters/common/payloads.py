from ...core.common import RuntimeOptionsView
from ...core.diagnostics import sanitize_json_object
from ...core.json_types import JsonValue


def raw_data_or_none(payload: dict[str, JsonValue], *, options: RuntimeOptionsView) -> dict[str, JsonValue] | None:
    """仅在启用时返回脱敏后的 raw_data。"""

    if not options.include_raw_data:
        return None
    return sanitize_json_object(payload)
