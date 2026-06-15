from ...core.common import ProviderRuntimeOptions
from ...core.diagnostics import sanitize_json_object


def raw_data_or_none(payload: dict, *, options: ProviderRuntimeOptions) -> dict | None:
    """仅在启用时返回脱敏后的 raw_data。"""

    if not options.include_raw_data:
        return None
    return sanitize_json_object(payload)
