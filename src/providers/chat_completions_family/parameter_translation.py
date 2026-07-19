from collections.abc import Mapping

from ...core.json_types import json_list_or_none
from ...i18n import translate
from ..common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    build_translation_context,
    normalize_positive_int,
    normalize_temperature,
    run_translators,
    set_target_value,
)
from .format import build_chat_response_format_payload


def translate_chat_temperature(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "temperature"), normalize_temperature(value))


def translate_chat_max_tokens(
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


def translate_chat_response_format(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    response_format = build_chat_response_format_payload(value, provider_label=context.provider_label)
    if response_format is None:
        return
    if "response_format" in envelope.body:
        raise ValueError(
            translate(
                "runtime.error.conflict",
                left="extra_params.body.response_format",
                right="response_format",
            )
        )
    envelope.body["response_format"] = response_format


def translate_chat_tools(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    existing_tools = json_list_or_none(envelope.body.get("tools")) or []
    incoming_tools = json_list_or_none(value)
    if incoming_tools is None:
        incoming_tools = [value]
    envelope.body["tools"] = [*existing_tools, *incoming_tools]


def translate_chat_identity(
    target_path: tuple[str, ...],
    *,
    field_name: str,
) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, target_path, value)

    _translator.__name__ = f"translate_chat_{field_name}"
    return _translator


CHAT_COMPLETIONS_FAMILY_TRANSLATORS: dict[str, FieldTranslator] = {
    "temperature": translate_chat_temperature,
    "max_tokens": translate_chat_max_tokens,
    "response_format": translate_chat_response_format,
    "top_p": translate_chat_identity(("body", "top_p"), field_name="top_p"),
    "tool_choice": translate_chat_identity(("body", "tool_choice"), field_name="tool_choice"),
    "tools": translate_chat_tools,
    "frequency_penalty": translate_chat_identity(("body", "frequency_penalty"), field_name="frequency_penalty"),
    "presence_penalty": translate_chat_identity(("body", "presence_penalty"), field_name="presence_penalty"),
    "seed": translate_chat_identity(("body", "seed"), field_name="seed"),
    "stop": translate_chat_identity(("body", "stop"), field_name="stop"),
    "n": translate_chat_identity(("body", "n"), field_name="n"),
}


def apply_chat_completions_family_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    *,
    extra_translators: Mapping[str, FieldTranslator] | None = None,
) -> None:
    """应用 Chat Completions 公共参数转译，可选注入 Provider 特有 translator。"""

    translators: dict[str, FieldTranslator] = dict(CHAT_COMPLETIONS_FAMILY_TRANSLATORS)
    if extra_translators is not None:
        translators.update(extra_translators)
    run_translators(context, envelope, translators)


__all__ = [
    "CHAT_COMPLETIONS_FAMILY_TRANSLATORS",
    "FieldTranslator",
    "TranslationContext",
    "TranslationEnvelope",
    "apply_chat_completions_family_parameters",
    "build_translation_context",
    "normalize_positive_int",
    "run_translators",
    "set_target_value",
]
