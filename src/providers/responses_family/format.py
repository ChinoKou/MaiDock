from ...schemas import (
    ObjectFields,
    OpenAITextConfig,
    OpenAITextFormatConfig,
    ResponseFormatSchemaSnapshot,
    ResponseRequestSnapshot,
)
from ..common.response_format import normalize_response_format_snapshot


def build_responses_text_format_payload(value: object, *, provider_label: str) -> dict | None:
    """将 Host response_format 转成 Responses text.format payload。"""

    if value is None:
        return None
    response_format = normalize_response_format_snapshot(value)
    format_type = response_format.format_type.strip().lower() if response_format.format_type is not None else None
    if format_type in {None, "text"}:
        return None
    if format_type in {"json_object", "json_obj"}:
        return {"type": "json_object"}
    if format_type != "json_schema":
        raise ValueError(f"{provider_label} 不支持的 response_format.format_type: {response_format.format_type}")
    schema_payload = response_format.schema_
    if isinstance(schema_payload, ObjectFields):
        raise ValueError(f"{provider_label} response_format=json_schema 需要提供 name 与 schema")
    if not isinstance(schema_payload, ResponseFormatSchemaSnapshot):
        raise ValueError(f"{provider_label} response_format=json_schema 需要提供 schema")
    if schema_payload.name is None or not schema_payload.name.strip():
        raise ValueError(f"{provider_label} response_format=json_schema 需要提供非空 name")
    if schema_payload.schema_ is None:
        raise ValueError(f"{provider_label} response_format=json_schema 需要提供 schema")
    payload: dict = {
        "type": "json_schema",
        "name": schema_payload.name.strip(),
        "schema": schema_payload.schema_.to_plain_dict(),
    }
    if schema_payload.description is not None:
        payload["description"] = schema_payload.description
    if schema_payload.strict is not None:
        payload["strict"] = schema_payload.strict
    return payload


def build_responses_text_config(
    request: ResponseRequestSnapshot,
) -> OpenAITextConfig | None:
    """将 Host response_format 转换为 OpenAI Responses text 配置。"""

    return build_responses_text_config_from_value(request.response_format)


def build_responses_text_config_from_value(
    response_format: object,
) -> OpenAITextConfig | None:
    """将 response_format 快照或值转换为 OpenAI Responses text 配置。"""

    format_payload = build_responses_text_format_payload(response_format, provider_label="OpenAI Responses")
    if format_payload is None:
        return None
    return OpenAITextConfig(format=OpenAITextFormatConfig.model_validate(format_payload))
