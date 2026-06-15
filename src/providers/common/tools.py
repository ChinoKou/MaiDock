from ...core.json_types import json_mapping_or_none
from ...core.parsing import ToolArgumentParseMode, fallback_tool_call_id, normalize_arguments
from ...schemas import ObjectFields


def resolve_tool_call_id(call_id: str | None, *, fallback_prefix: str, index: int) -> str:
    normalized_call_id = call_id.strip() if isinstance(call_id, str) else ""
    if normalized_call_id:
        return normalized_call_id
    return fallback_tool_call_id(f"{fallback_prefix}_{index}")


def normalize_tool_arguments_value(arguments: object, parse_mode: ToolArgumentParseMode) -> dict:
    argument_mapping = json_mapping_or_none(arguments)
    if argument_mapping is not None:
        return ObjectFields.from_unknown(argument_mapping).to_plain_dict()
    return normalize_arguments(arguments if isinstance(arguments, str) else None, parse_mode)
