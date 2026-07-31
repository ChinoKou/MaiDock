from ...core.common import RuntimeOptionsView, build_usage_from_snapshot, read_model_identifier
from ...core.diagnostics import sanitize_json_object
from ...core.json_types import JsonValue, json_mapping_or_none, mapping_field, mapping_to_json_object
from ...core.parameter_catalog import get_parameter_catalog
from ...i18n import runtime_expected, runtime_item, translate
from ...schemas import AudioTranscriptionRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..chat_completions_family.audio import (
    AudioFormat,
    build_chat_audio_message,
    prepare_chat_audio,
)
from ..chat_completions_family.parameter_translation import (
    TranslationEnvelope,
    build_translation_context,
)
from .parameter_translation import apply_mimo_audio_parameters

MIMO_AUDIO_TRANSCRIPTION_LABEL = "Xiaomi Mimo Audio Transcription"
_MIMO_AUDIO_FORMATS: frozenset[AudioFormat] = frozenset({"mp3", "wav"})
_MIMO_AUDIO_MAX_BASE64_CHARS = 10 * 1024 * 1024
_MIMO_AUDIO_UNSUPPORTED_BODY_FIELDS = frozenset(
    {
        "frequency_penalty",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "prompt",
        "response_format",
        "seed",
        "stop",
        "stream_options",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_p",
    }
)


def build_mimo_audio_transcription_request(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: RuntimeOptionsView,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue]]:
    """构建 Mimo ASR 请求。"""

    model = read_model_identifier(request.model_info)
    overrides = options.parameter_overrides.get("xiaomi_mimo", "audio_transcription")
    catalog = get_parameter_catalog("xiaomi_mimo", "audio_transcription")
    context = build_translation_context(
        request,
        overrides=overrides,
        catalog=catalog,
        provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
        provider="xiaomi_mimo",
        capability="audio_transcription",
        model=model,
    )
    envelope = TranslationEnvelope(body={"model": model, "stream": False})
    apply_mimo_audio_parameters(context, envelope)
    body = dict(envelope.body)
    format_hints = {
        "format": body.pop("format", None),
        "audio_format": body.pop("audio_format", None),
    }

    audio = prepare_chat_audio(
        request.audio_base64,
        format_hints,
        provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
        allowed_formats=_MIMO_AUDIO_FORMATS,
        max_base64_chars=_MIMO_AUDIO_MAX_BASE64_CHARS,
    )
    raw_asr_options_value = body.pop("asr_options", None)
    raw_asr_options = json_mapping_or_none(raw_asr_options_value)
    if raw_asr_options_value is not None and raw_asr_options is None:
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="Mimo asr_options",
                expected=runtime_expected("object"),
                actual=type(raw_asr_options_value).__name__,
            )
        )
    asr_options = mapping_to_json_object(raw_asr_options) if raw_asr_options is not None else {}
    language = asr_options.get("language", "auto")
    if language not in {"auto", "zh", "en"}:
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="Mimo asr_options.language",
                allowed="auto/zh/en",
            )
        )
    asr_options["language"] = language
    for field_name in _MIMO_AUDIO_UNSUPPORTED_BODY_FIELDS:
        body.pop(field_name, None)
    body["asr_options"] = asr_options
    body["messages"] = [build_chat_audio_message(audio, include_format=True)]

    body["model"] = model
    body["stream"] = False
    return body, envelope.headers, envelope.query


def parse_mimo_audio_transcription_response(
    payload: dict[str, JsonValue], *, options: RuntimeOptionsView
) -> ProviderResponse:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            translate(
                "runtime.error.response_missing",
                provider=MIMO_AUDIO_TRANSCRIPTION_LABEL,
                item="choices",
            )
        )
    first = json_mapping_or_none(choices[0])
    if first is None:
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=f"{MIMO_AUDIO_TRANSCRIPTION_LABEL} choices[0]",
                expected=runtime_expected("object"),
                actual=type(choices[0]).__name__,
            )
        )
    message = mapping_field(first, "message")
    content: str | None = None
    if message is not None:
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                item_mapping = json_mapping_or_none(item)
                if item_mapping is not None:
                    text = item_mapping.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
            content = "".join(parts) or None
    if not content:
        raise ValueError(
            translate(
                "runtime.error.response_missing",
                provider=MIMO_AUDIO_TRANSCRIPTION_LABEL,
                item=runtime_item("audio_transcription_text"),
            )
        )
    return ProviderResponse(
        content=content,
        usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
        raw_data=sanitize_json_object(payload) if options.include_raw_data else None,
    )
