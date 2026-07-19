from ...core.json_types import json_mapping_or_none
from ...i18n import translate
from ..common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    normalize_positive_int,
    normalize_temperature,
    run_translators,
    set_target_value,
)


def translate_anthropic_temperature(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "temperature"), normalize_temperature(value))


def translate_anthropic_max_tokens(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(
        envelope,
        ("body", "max_tokens"),
        normalize_positive_int(value, field_name="max_tokens"),
    )


def translate_anthropic_identity(target_path: tuple[str, ...], *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, target_path, value)

    _translator.__name__ = f"translate_anthropic_{field_name}"
    return _translator


def reject_anthropic_response_format_params(context: TranslationContext) -> None:
    request = context.request
    if request is None:
        return
    response_format = getattr(request, "response_format", None)
    if response_format is not None:
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="Anthropic Messages request.response_format",
                allowed="unset",
            )
        )
    for source_name, source in (
        ("model_info.extra_params", request.model_info.extra_params.fields),
        ("request.extra_params", request.extra_params.fields),
    ):
        if "response_format" in source:
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject=f"Anthropic Messages {source_name}.response_format",
                    allowed="unset",
                )
            )
        body = json_mapping_or_none(source.get("body"))
        if body is not None and "response_format" in body:
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject=f"Anthropic Messages {source_name}.body.response_format",
                    allowed="unset",
                )
            )


ANTHROPIC_TRANSLATORS: dict[str, FieldTranslator] = {
    "temperature": translate_anthropic_temperature,
    "max_tokens": translate_anthropic_max_tokens,
    "top_p": translate_anthropic_identity(("body", "top_p"), field_name="top_p"),
    "top_k": translate_anthropic_identity(("body", "top_k"), field_name="top_k"),
    "thinking": translate_anthropic_identity(("body", "thinking"), field_name="thinking"),
    "tool_choice": translate_anthropic_identity(("body", "tool_choice"), field_name="tool_choice"),
    "stop_sequences": translate_anthropic_identity(("body", "stop_sequences"), field_name="stop_sequences"),
    "metadata": translate_anthropic_identity(("body", "metadata"), field_name="metadata"),
    "service_tier": translate_anthropic_identity(("body", "service_tier"), field_name="service_tier"),
}


def apply_anthropic_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    reject_anthropic_response_format_params(context)
    run_translators(context, envelope, ANTHROPIC_TRANSLATORS)
