from ...i18n import runtime_item, translate
from ...schemas import ResponseFormatSchemaSnapshot
from ..common.response_format import normalize_response_format_snapshot
from ...core.json_types import JsonValue


def build_chat_response_format_payload(value: object, *, provider_label: str) -> dict[str, JsonValue] | None:
    """将 Host response_format 转成 Chat Completions payload。"""

    if value is None:
        return None
    response_format = normalize_response_format_snapshot(value)
    format_type = response_format.format_type.strip().lower() if response_format.format_type is not None else None
    if format_type in {None, "text"}:
        return None
    if format_type in {"json_object", "json_obj"}:
        return {"type": "json_object"}
    if format_type != "json_schema":
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject=f"{provider_label} response_format.format_type",
                allowed="text/json_object/json_schema",
            )
        )
    schema_payload = response_format.schema_
    if not isinstance(schema_payload, ResponseFormatSchemaSnapshot):
        raise ValueError(
            translate(
                "runtime.error.required",
                subject=f"{provider_label} json_schema",
                field=runtime_item("name_and_schema"),
            )
        )
    if schema_payload.name is None or not schema_payload.name.strip():
        raise ValueError(
            translate(
                "runtime.error.required",
                subject=f"{provider_label} json_schema",
                field=runtime_item("non_empty_name"),
            )
        )
    if schema_payload.schema_ is None:
        raise ValueError(translate("runtime.error.required", subject=f"{provider_label} json_schema", field="schema"))
    json_schema: dict[str, JsonValue] = {
        "name": schema_payload.name.strip(),
        "schema": schema_payload.schema_.to_plain_dict(),
    }
    if schema_payload.description is not None:
        json_schema["description"] = schema_payload.description
    if schema_payload.strict is not None:
        json_schema["strict"] = schema_payload.strict
    return {"type": "json_schema", "json_schema": json_schema}


def build_chat_response_format_body(response_format: object) -> dict[str, JsonValue] | None:
    """将 Host response_format 转换为 Chat Completions response_format body。"""
    return build_chat_response_format_payload(response_format, provider_label="Chat Completions")
