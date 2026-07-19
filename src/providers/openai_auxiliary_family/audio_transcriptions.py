import json
from collections.abc import Callable

import httpx

from ...core.common import ProviderRuntimeOptions, build_audio_file, read_model_identifier
from ...core.diagnostics import build_parse_error_message, sanitize_for_log, sanitize_json_object
from ...core.json_types import json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import ProviderPolicyKey, apply_transport_parameter_policy
from ...i18n import runtime_item, translate
from ...schemas import AudioTranscriptionRequestSnapshot
from .parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    build_translation_context,
)

type AudioParameterApplier = Callable[[TranslationContext, TranslationEnvelope], None]


def form_field_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_multipart_audio_transcription_request(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
    provider_label: str,
    policy_provider: ProviderPolicyKey,
    apply_parameters: AudioParameterApplier,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]], dict[str, str], dict]:
    """构建 OpenAI 兼容的 multipart 音频转录请求。"""

    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get(policy_provider, "audio_transcription")
    catalog = get_parameter_catalog(policy_provider, "audio_transcription")
    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=provider_label,
        provider=policy_provider,
        capability="audio_transcription",
        model=model,
    )
    envelope = TranslationEnvelope(body={"model": model})
    apply_parameters(context, envelope)

    audio_filename, audio_buffer = build_audio_file(request)
    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=provider_label,
        capability="audio_transcription",
    )
    form_data = {str(key): form_field_value(value) for key, value in transport.body.items() if key != "file"}
    return (
        form_data,
        {"file": (audio_filename, audio_buffer.getvalue())},
        transport.headers,
        transport.query,
    )


def parse_multipart_audio_transcription_response(
    response: httpx.Response,
    *,
    options: ProviderRuntimeOptions,
    provider_label: str,
) -> tuple[str, dict | None]:
    """解析 OpenAI 兼容的音频转录文本响应。"""

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
    message = translate("runtime.error.output_missing", item=runtime_item("text_content"))
    raise ValueError(build_parse_error_message(provider_label, message))
