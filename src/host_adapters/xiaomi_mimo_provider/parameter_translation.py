from ...core.json_types import JsonValue
from ...i18n import runtime_expected, runtime_subject, translate
from ..chat_completions_family.parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    apply_chat_completions_family_parameters,
    normalize_positive_int,
    run_translators,
    set_target_value,
)

MIMO_MAX_COMPLETION_TOKENS = 131072


def _normalize_mimo_max_tokens(value: object) -> int:
    normalized = normalize_positive_int(value, field_name="max_completion_tokens")
    if normalized > MIMO_MAX_COMPLETION_TOKENS:
        raise ValueError(
            translate(
                "runtime.error.limit",
                subject="max_completion_tokens",
                limit=MIMO_MAX_COMPLETION_TOKENS,
            )
        )
    return normalized


def _translate_mimo_max_tokens(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    normalized = _normalize_mimo_max_tokens(value)
    set_target_value(
        envelope,
        ("body", "max_completion_tokens"),
        normalized,
    )


def _translate_mimo_thinking(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "thinking"), value)


def _translate_mimo_audio_language(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    if value not in {"auto", "zh", "en"}:
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="Mimo asr_options.language",
                allowed="auto/zh/en",
            )
        )
    set_target_value(envelope, ("body", "asr_options", "language"), value)


def _translate_mimo_audio_format(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "format"), value)


def _translate_mimo_audio_format_alias(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "audio_format"), value)


def apply_mimo_chat_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
) -> None:
    apply_chat_completions_family_parameters(
        context,
        envelope,
        extra_translators={
            "max_tokens": _translate_mimo_max_tokens,
            "thinking": _translate_mimo_thinking,
        },
    )


def apply_mimo_audio_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
) -> None:
    run_translators(
        context,
        envelope,
        {
            "audio_format": _translate_mimo_audio_format_alias,
            "format": _translate_mimo_audio_format,
            "language": _translate_mimo_audio_language,
        },
    )


def mimo_thinking_enabled(body: dict[str, JsonValue]) -> bool:
    """按 Mimo 官方 thinking.type 规则判断本次请求是否启用思考。"""

    thinking = body.get("thinking")
    if thinking is None:
        return True
    if not isinstance(thinking, dict):
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject=runtime_subject("mimo_thinking"),
                expected=runtime_expected("object"),
                actual=type(thinking).__name__,
            )
        )
    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        return False
    if thinking_type == "enabled":
        return True
    raise ValueError(
        translate(
            "runtime.error.unsupported_value",
            subject="Mimo thinking.type",
            allowed="enabled/disabled",
        )
    )
