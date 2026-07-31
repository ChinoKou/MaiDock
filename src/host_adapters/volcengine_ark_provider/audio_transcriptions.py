import logging

from ...core.common import RuntimeOptionsView, read_model_identifier
from ...core.parameter_catalog import get_parameter_catalog
from ...i18n import runtime_subject, translate
from ...schemas import AudioTranscriptionRequestSnapshot, ProviderResponse
from ..responses_family.audio import (
    AudioFormat,
    build_responses_audio_input,
    parse_responses_audio_transcription,
)
from ..responses_family.parameter_translation import (
    TranslationEnvelope,
    build_translation_context,
)
from .parameter_translation import apply_ark_audio_parameters
from .responses import VOLCENGINE_PROVIDER_LABEL, build_ark_request_headers, create_responses_mapper
from ...core.json_types import JsonValue, json_array

ARK_AUDIO_TRANSCRIPTION_LABEL = "Volcengine Ark Audio Transcription"
_ARK_AUDIO_FORMATS: frozenset[AudioFormat] = frozenset({"mp3", "wav", "aac", "m4a"})
_ARK_AUDIO_MAX_DECODED_BYTES = 25 * 1024 * 1024


def build_ark_audio_transcription_request(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: RuntimeOptionsView,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue]]:
    """构建 ARK Responses input_audio 语音转录请求。"""

    model = read_model_identifier(request.model_info)
    overrides = options.parameter_overrides.get("volcengine_ark", "audio_transcription")
    catalog = get_parameter_catalog("volcengine_ark", "audio_transcription")
    context = build_translation_context(
        request,
        overrides=overrides,
        catalog=catalog,
        provider_label=ARK_AUDIO_TRANSCRIPTION_LABEL,
        provider="volcengine_ark",
        capability="audio_transcription",
        model=model,
    )
    envelope = TranslationEnvelope(body={"model": model, "stream": False})
    apply_ark_audio_parameters(context, envelope)
    body = dict(envelope.body)
    format_hints = {
        "format": body.pop("format", None),
        "audio_format": body.pop("audio_format", None),
    }
    configured_prompt = body.pop("prompt", None)
    prompt = configured_prompt if configured_prompt is not None else "请识别音频中的内容，以文字形式返回识别结果。"
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            translate(
                "runtime.error.required",
                subject=runtime_subject("ark_settings"),
                field="audio_transcription_prompt",
            )
        )
    for unsupported_key in ("caching", "instructions", "previous_response_id", "tools"):
        body.pop(unsupported_key, None)
    body["model"] = model
    body["stream"] = False
    body["input"] = json_array(
        build_responses_audio_input(
            request.audio_base64,
            format_hints,
            prompt=prompt.strip(),
            provider_label=ARK_AUDIO_TRANSCRIPTION_LABEL,
            allowed_formats=_ARK_AUDIO_FORMATS,
            max_decoded_bytes=_ARK_AUDIO_MAX_DECODED_BYTES,
        )
    )
    headers = build_ark_request_headers(envelope.headers, body)
    return body, headers, envelope.query


def parse_ark_audio_transcription_response(
    payload: dict[str, JsonValue],
    *,
    options: RuntimeOptionsView,
) -> ProviderResponse:
    """使用 Responses Family 解析器提取 ARK 转录文本和 usage。"""

    mapper = create_responses_mapper(options=options, logger=logging.getLogger(__name__))
    return parse_responses_audio_transcription(
        payload,
        mapper=mapper,
        provider_label=VOLCENGINE_PROVIDER_LABEL,
    )
