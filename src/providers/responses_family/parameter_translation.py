from typing import cast

from ...providers.common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    merge_body_object,
    normalize_json_object_value,
    normalize_positive_int,
    normalize_temperature,
    run_translators,
    set_target_value,
)
from ..common.response_format import build_responses_text_format_payload


def translate_response_temperature(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "temperature"), normalize_temperature(value))


def translate_response_max_output_tokens(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(
        envelope,
        ("body", "max_output_tokens"),
        normalize_positive_int(value, field_name="max_tokens"),
    )


def translate_response_text_format(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    format_payload = build_responses_text_format_payload(value, provider_label=context.provider_label)
    if format_payload is None:
        return
    text_object = merge_body_object(envelope, "text", {})
    if "format" in text_object:
        raise ValueError("extra_params.text.format 与 response_format 不能同时设置")
    text_object["format"] = format_payload


def translate_response_tools(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    if value is None:
        return
    current_value = envelope.body.get("tools")
    current_tools: list[object] = cast(list[object], current_value) if isinstance(current_value, list) else []
    incoming_tools: list[object] = cast(list[object], value) if isinstance(value, list) else [value]
    envelope.body["tools"] = [*current_tools, *incoming_tools]


def translate_response_text_object(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    text_object = normalize_json_object_value(value, field_name="text")
    existing = merge_body_object(envelope, "text", {})
    existing.update(text_object)


def translate_response_identity(
    target_path: tuple[str, ...],
    *,
    field_name: str,
) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, target_path, value)

    _translator.__name__ = f"translate_{field_name}"
    return _translator


RESPONSES_TRANSLATORS: dict[str, FieldTranslator] = {
    "temperature": translate_response_temperature,
    "max_tokens": translate_response_max_output_tokens,
    "response_format": translate_response_text_format,
    "reasoning": translate_response_identity(("body", "reasoning"), field_name="reasoning"),
    "thinking": translate_response_identity(("body", "thinking"), field_name="thinking"),
    "text": translate_response_text_object,
    "tool_choice": translate_response_identity(("body", "tool_choice"), field_name="tool_choice"),
    "parallel_tool_calls": translate_response_identity(
        ("body", "parallel_tool_calls"),
        field_name="parallel_tool_calls",
    ),
    "max_tool_calls": translate_response_identity(("body", "max_tool_calls"), field_name="max_tool_calls"),
    "include": translate_response_identity(("body", "include"), field_name="include"),
    "instructions": translate_response_identity(("body", "instructions"), field_name="instructions"),
    "metadata": translate_response_identity(("body", "metadata"), field_name="metadata"),
    "store": translate_response_identity(("body", "store"), field_name="store"),
    "truncation": translate_response_identity(("body", "truncation"), field_name="truncation"),
    "service_tier": translate_response_identity(("body", "service_tier"), field_name="service_tier"),
    "previous_response_id": translate_response_identity(
        ("body", "previous_response_id"),
        field_name="previous_response_id",
    ),
    "user": translate_response_identity(("body", "user"), field_name="user"),
    "session": translate_response_identity(("body", "session"), field_name="session"),
    "caching": translate_response_identity(("body", "caching"), field_name="caching"),
    "expire_at": translate_response_identity(("body", "expire_at"), field_name="expire_at"),
    "top_p": translate_response_identity(("body", "top_p"), field_name="top_p"),
    "tools": translate_response_tools,
}


def apply_responses_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
) -> None:
    run_translators(context, envelope, RESPONSES_TRANSLATORS)
