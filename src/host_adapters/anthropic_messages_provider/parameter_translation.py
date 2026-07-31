from ...i18n import translate
from ...schemas import ResponseRequestSnapshot
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


def reject_anthropic_response_format_params(request: ResponseRequestSnapshot) -> None:
    # Anthropic 只拒绝 Core 类型化的 response_format；覆写目录不开放该字段。
    if request.response_format is not None:
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="Anthropic Messages request.response_format",
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
    run_translators(context, envelope, ANTHROPIC_TRANSLATORS)
