import httpx

from ...core.common import ProviderRuntimeOptions, build_usage_from_snapshot, read_model_identifier
from ...core.diagnostics import sanitize_json_object
from ...core.json_types import json_mapping_or_none, mapping_field, mapping_to_json_object
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...i18n import runtime_expected, runtime_item, runtime_subject, translate
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
from ..chat_completions_family.transport import create_async_client, post_json
from .chat import MIMO_CHAT_COMPLETIONS_ENDPOINT, build_client_config, resolve_path
from .parameter_translation import apply_mimo_audio_parameters, normalize_mimo_chat_body

MIMO_ASR_MODEL = "mimo-v2.5-asr"
MIMO_AUDIO_TRANSCRIPTION_LABEL = "Xiaomi Mimo Audio Transcription"
_MIMO_ASR_FORMATS: frozenset[AudioFormat] = frozenset({"mp3", "wav"})
_MIMO_GENERIC_AUDIO_FORMATS: frozenset[AudioFormat] = frozenset({"mp3", "wav", "flac", "m4a", "ogg"})
_MIMO_ASR_MAX_BASE64_CHARS = 10 * 1024 * 1024
_MIMO_GENERIC_MAX_BASE64_CHARS = 50 * 1024 * 1024
_MIMO_ASR_UNSUPPORTED_BODY_FIELDS = frozenset(
    {
        "frequency_penalty",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
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
    options: ProviderRuntimeOptions,
) -> tuple[dict, dict[str, str], dict]:
    """按模型构建 Mimo 专用 ASR 或通用音频理解请求。"""

    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get("xiaomi_mimo", "audio_transcription")
    catalog = get_parameter_catalog("xiaomi_mimo", "audio_transcription")
    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
        provider="xiaomi_mimo",
        capability="audio_transcription",
        model=model,
    )
    envelope = TranslationEnvelope(body={"model": model, "stream": False})
    apply_mimo_audio_parameters(context, envelope)
    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
        capability="audio_transcription",
    )
    body = dict(transport.body)
    normalize_mimo_chat_body(body)
    format_hints = {
        "format": body.pop("format", None),
        "audio_format": body.pop("audio_format", None),
    }
    configured_prompt = body.pop("prompt", None)

    if model == MIMO_ASR_MODEL:
        audio = prepare_chat_audio(
            request.audio_base64,
            format_hints,
            provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
            allowed_formats=_MIMO_ASR_FORMATS,
            max_base64_chars=_MIMO_ASR_MAX_BASE64_CHARS,
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
        language = options.mimo_audio_transcription_language
        if "language" in asr_options:
            language = asr_options["language"]
        if language not in {"auto", "zh", "en"}:
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject="Mimo asr_options.language",
                    allowed="auto/zh/en",
                )
            )
        asr_options["language"] = language
        for field_name in _MIMO_ASR_UNSUPPORTED_BODY_FIELDS:
            body.pop(field_name, None)
        body["asr_options"] = asr_options
        body["messages"] = [build_chat_audio_message(audio, include_format=True)]
    else:
        prompt = configured_prompt if configured_prompt is not None else options.mimo_audio_transcription_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                translate(
                    "runtime.error.required",
                    subject=runtime_subject("mimo_settings"),
                    field="audio_transcription_prompt",
                )
            )
        audio = prepare_chat_audio(
            request.audio_base64,
            format_hints,
            provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
            allowed_formats=_MIMO_GENERIC_AUDIO_FORMATS,
            max_base64_chars=_MIMO_GENERIC_MAX_BASE64_CHARS,
        )
        body.pop("asr_options", None)
        body["messages"] = [build_chat_audio_message(audio, prompt=prompt.strip())]
        if options.mimo_force_disable_thinking:
            body["thinking"] = {"type": "disabled"}

    body["model"] = model
    body["stream"] = False
    return body, transport.headers, transport.query


async def build_mimo_audio_transcription(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProviderResponse:
    body, extra_headers, extra_query = build_mimo_audio_transcription_request(request, options=options)
    config = build_client_config(
        request.api_provider,
        user_agent=options.mimo_user_agent,
        default_max_retries=options.mimo_max_retries,
        force_max_retries=options.mimo_force_max_retries,
        default_retry_interval=options.mimo_retry_interval,
        force_retry_interval=options.mimo_force_retry_interval,
    )
    path = resolve_path(config, MIMO_CHAT_COMPLETIONS_ENDPOINT)
    async with create_async_client(config, transport=transport) as client:
        payload = await post_json(
            client,
            path,
            json_body=body,
            headers=extra_headers,
            query=extra_query,
            provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
            max_retries=config.max_retries,
            retry_interval=config.retry_interval,
        )
    return _parse_audio_transcription_response(payload, options=options)


def _parse_audio_transcription_response(payload: dict, *, options: ProviderRuntimeOptions) -> ProviderResponse:
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
