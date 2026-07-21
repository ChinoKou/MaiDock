from ...core.common import ProviderRuntimeOptions, read_model_identifier
from ...core.diagnostics import build_parse_error_message, sanitize_json_object
from ...core.json_types import json_mapping_or_none, list_field, mapping_field, string_field
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...i18n import runtime_item, translate
from ...schemas import AudioTranscriptionRequestSnapshot
from ..common.audio import AudioFormat, prepare_base64_audio
from ..common.parameter_translation import TranslationEnvelope, build_translation_context
from .chat import DASHSCOPE_PROVIDER_LABEL
from .errors import raise_for_dashscope_error
from .parameter_translation import apply_dashscope_audio_parameters

_DASHSCOPE_ASR_FORMATS: frozenset[AudioFormat] = frozenset({"wav", "mp3", "aac", "flac", "ogg"})
_DASHSCOPE_ASR_MAX_BASE64_CHARS = 10 * 1024 * 1024


def build_audio_transcription_request(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[dict, dict[str, str], dict]:
    """构建用于音频转录的阿里云百炼 DashScope 多模态生成请求。"""
    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get("dashscope", "audio_transcription")
    catalog = get_parameter_catalog("dashscope", "audio_transcription")

    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=DASHSCOPE_PROVIDER_LABEL,
        provider="dashscope",
        capability="audio_transcription",
        model=model,
    )
    envelope = TranslationEnvelope(body={"model": model})
    apply_dashscope_audio_parameters(context, envelope)

    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=DASHSCOPE_PROVIDER_LABEL,
        capability="audio_transcription",
    )

    body = dict(transport.body)
    format_hints = {
        "format": body.pop("format", None),
        "audio_format": body.pop("audio_format", None),
    }
    audio = prepare_base64_audio(
        request.audio_base64,
        format_hints,
        provider_label=f"{DASHSCOPE_PROVIDER_LABEL} Audio Transcriptions",
        allowed_formats=_DASHSCOPE_ASR_FORMATS,
        max_base64_chars=_DASHSCOPE_ASR_MAX_BASE64_CHARS,
    )
    parameters = json_mapping_or_none(body.get("parameters"))
    if parameters is not None:
        parameters = dict(parameters)
    else:
        parameters = {}
    parameters.setdefault("result_format", "message")
    body["parameters"] = parameters
    body["input"] = {"messages": [{"role": "user", "content": [{"audio": audio.data_url}]}]}

    return body, transport.headers, transport.query


def parse_audio_transcription_response(
    payload: dict,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[str, dict | None]:
    """解析阿里云百炼 DashScope 多模态生成响应，提取转录文本。"""
    raise_for_dashscope_error(payload)
    output = mapping_field(payload, "output")
    choices = list_field(output, "choices") if output is not None else None
    first_choice = json_mapping_or_none(choices[0]) if choices else None
    message = mapping_field(first_choice, "message") if first_choice is not None else None
    content = message.get("content") if message is not None else None
    text: str | None = None
    if isinstance(content, str) and content:
        text = content
    elif isinstance(content, list):
        # 多模态端点的 content 可能是 [{"text": "..."}] 列表格式
        parts: list[str] = []
        for item in content:
            item_mapping = json_mapping_or_none(item)
            if item_mapping is None:
                continue
            item_text = string_field(item_mapping, "text")
            if item_text:
                parts.append(item_text)
        text = "\n".join(parts) if parts else None
    if text:
        raw_data = sanitize_json_object(payload) if options.include_raw_data else None
        return text, raw_data
    provider_label = f"{DASHSCOPE_PROVIDER_LABEL} Audio Transcriptions"
    message = translate("runtime.error.output_missing", item=runtime_item("text_content"))
    raise ValueError(build_parse_error_message(provider_label, message))
