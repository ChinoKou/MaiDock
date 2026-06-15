import json

import httpx

from ...core.common import ProviderRuntimeOptions, build_audio_file, read_model_identifier
from ...core.diagnostics import build_parse_error_message, sanitize_for_log, sanitize_json_object
from ...core.json_types import json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...schemas import AudioTranscriptionRequestSnapshot
from ..common.parameter_translation import build_translation_context, TranslationEnvelope
from .parameter_translation import apply_openai_audio_parameters
from .responses import OPENAI_PROVIDER_LABEL


def _form_field_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_audio_transcription_request(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]], dict[str, str], dict]:
    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get("openai_responses", "audio_transcription")
    catalog = get_parameter_catalog("openai_responses", "audio_transcription")

    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=OPENAI_PROVIDER_LABEL,
        provider="openai_responses",
        capability="audio_transcription",
        model=model,
    )
    envelope = TranslationEnvelope(body={"model": model})
    apply_openai_audio_parameters(context, envelope)

    audio_filename, audio_buffer = build_audio_file(request)
    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=OPENAI_PROVIDER_LABEL,
        capability="audio_transcription",
    )

    form_data = {str(k): _form_field_value(v) for k, v in transport.body.items() if k != "file"}
    return (
        form_data,
        {"file": (audio_filename, audio_buffer.getvalue())},
        transport.headers,
        transport.query,
    )


def parse_audio_transcription_response(
    response: httpx.Response,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[str, dict | None]:
    payload: object | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    payload_mapping = json_mapping_or_none(payload)
    if payload_mapping is not None:
        text = payload_mapping.get("text")
        if isinstance(text, str):
            raw_data = sanitize_json_object(payload_mapping) if options.include_raw_data else None
            return text, raw_data
    if isinstance(payload, str):
        raw_data = {"text": sanitize_for_log(payload)} if options.include_raw_data else None
        return payload, raw_data
    content = response.text
    if content:
        raw_data = {"text": sanitize_for_log(content)} if options.include_raw_data else None
        return content, raw_data
    raise ValueError(build_parse_error_message("OpenAI Audio Transcriptions", "缺少文本内容"))
