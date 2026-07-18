from ...core.json_types import json_mapping_or_none, mapping_to_json_object
from ...schemas import AudioTranscriptionRequestSnapshot, ResponseRequestSnapshot
from ..chat_completions_family.parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    apply_chat_completions_family_parameters,
    normalize_positive_int,
    set_target_value,
)

MIMO_MAX_COMPLETION_TOKENS = 131072


def _normalize_mimo_max_tokens(value: object) -> int:
    normalized = normalize_positive_int(value, field_name="max_completion_tokens")
    if normalized > MIMO_MAX_COMPLETION_TOKENS:
        raise ValueError(f"max_completion_tokens 不能超过 {MIMO_MAX_COMPLETION_TOKENS}")
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


def _explicit_mimo_max_token_values(context: TranslationContext) -> list[tuple[str, int]]:
    request = context.request
    if request is None:
        return []
    candidates: list[tuple[str, object]] = []
    if request.model_info.max_tokens is not None:
        candidates.append(("model_info.max_tokens", request.model_info.max_tokens))
    if isinstance(request, (ResponseRequestSnapshot, AudioTranscriptionRequestSnapshot)):
        if request.max_tokens is not None:
            candidates.append(("request.max_tokens", request.max_tokens))
    extra_sources: list[tuple[str, dict]] = []
    if context.policy.accept_model_extra_params:
        extra_sources.append(("model_info.extra_params", request.model_info.extra_params.fields))
    if context.policy.accept_request_extra_params:
        extra_sources.append(("request.extra_params", request.extra_params.fields))
    for source_name, source in extra_sources:
        for field_name in ("max_tokens", "max_completion_tokens"):
            value = source.get(field_name)
            if value is not None:
                candidates.append((f"{source_name}.{field_name}", value))
        body = json_mapping_or_none(source.get("body"))
        if body is None:
            continue
        for field_name in ("max_tokens", "max_completion_tokens"):
            value = body.get(field_name)
            if value is not None:
                candidates.append((f"{source_name}.body.{field_name}", value))
    return [(source, _normalize_mimo_max_tokens(value)) for source, value in candidates]


def _canonicalize_mimo_max_tokens(context: TranslationContext) -> None:
    """校验显式 Host 来源，并把旧 body 字段纳入统一参数管线。"""

    raw_body = json_mapping_or_none(context.normalized.fields.get("body"))
    legacy_value: object | None = None
    official_value: object | None = None
    if raw_body is not None:
        body = mapping_to_json_object(raw_body)
        legacy_value = body.pop("max_tokens", None)
        official_value = body.pop("max_completion_tokens", None)
        context.normalized.fields["body"] = body

    target_path = "body.max_completion_tokens"
    legacy_target_path = "body.max_tokens"
    disabled_paths = set(context.policy.disabled_paths)
    if target_path in disabled_paths or legacy_target_path in disabled_paths:
        context.normalized.fields.pop("max_tokens", None)
        context.normalized.sources.pop("max_tokens", None)
        return

    explicit_values = _explicit_mimo_max_token_values(context)
    if legacy_target_path in context.policy.rejected_paths and (
        explicit_values
        or "max_tokens" in context.normalized.fields
        or legacy_value is not None
        or official_value is not None
    ):
        raise ValueError(f"{context.provider_label} {context.capability} 参数策略拒绝路径: {legacy_target_path}")
    if explicit_values:
        expected_source, expected_value = explicit_values[0]
        for source, value in explicit_values[1:]:
            if value != expected_value:
                raise ValueError(f"{expected_source} 与 {source} 同时提供了不同的 max_tokens/max_completion_tokens")
        context.normalized.fields["max_tokens"] = expected_value
        context.normalized.sources["max_tokens"] = expected_source
        return

    if "max_tokens" in context.normalized.fields:
        return
    candidate = official_value if official_value is not None else legacy_value
    if legacy_value is not None and official_value is not None:
        normalized_legacy = _normalize_mimo_max_tokens(legacy_value)
        normalized_official = _normalize_mimo_max_tokens(official_value)
        if normalized_legacy != normalized_official:
            raise ValueError("body.max_tokens 与 body.max_completion_tokens 不能同时设置不同值")
        candidate = normalized_official
    if candidate is not None:
        context.normalized.fields["max_tokens"] = _normalize_mimo_max_tokens(candidate)


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
        raise ValueError("Mimo asr_options.language 仅支持 auto、zh 或 en")
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


def _translate_mimo_audio_prompt(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "prompt"), value)


def apply_mimo_chat_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
) -> None:
    _canonicalize_mimo_max_tokens(context)
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
    _canonicalize_mimo_max_tokens(context)
    apply_chat_completions_family_parameters(
        context,
        envelope,
        extra_translators={
            "audio_format": _translate_mimo_audio_format_alias,
            "format": _translate_mimo_audio_format,
            "language": _translate_mimo_audio_language,
            "max_tokens": _translate_mimo_max_tokens,
            "prompt": _translate_mimo_audio_prompt,
        },
    )


def normalize_mimo_chat_body(body: dict) -> None:
    """兼容 transport override 中的旧 body.max_tokens。"""

    legacy_value = body.pop("max_tokens", None)
    official_value = body.get("max_completion_tokens")
    if legacy_value is not None:
        normalized_legacy = _normalize_mimo_max_tokens(legacy_value)
        body["max_completion_tokens"] = normalized_legacy
    elif official_value is not None:
        body["max_completion_tokens"] = _normalize_mimo_max_tokens(official_value)


def mimo_thinking_enabled(body: dict) -> bool:
    """按 Mimo 官方 thinking.type 规则判断本次请求是否启用思考。"""

    thinking = body.get("thinking")
    if thinking is None:
        return True
    if not isinstance(thinking, dict):
        raise TypeError("Mimo thinking 必须是 object")
    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        return False
    if thinking_type == "enabled":
        return True
    raise ValueError("Mimo thinking.type 仅支持 enabled 或 disabled")
