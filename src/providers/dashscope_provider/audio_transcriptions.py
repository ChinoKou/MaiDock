import base64

from ...core.common import (
    ProviderRuntimeOptions,
    build_audio_file,
    read_model_identifier,
)
from ...core.diagnostics import build_parse_error_message, sanitize_json_object
from ...core.json_types import json_mapping_or_none, list_field, mapping_field
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...schemas import AudioTranscriptionRequestSnapshot
from ..common.parameter_translation import (
    build_translation_context,
    TranslationEnvelope,
)
from .chat import DASHSCOPE_PROVIDER_LABEL
from .parameter_translation import apply_dashscope_audio_parameters

# 常见音频格式魔数 → MIME 类型映射
_AUDIO_MIME_BY_MAGIC: dict[bytes, str] = {
    b"RIFF": "audio/wav",
    b"ID3": "audio/mpeg",
    b"\xff\xfb": "audio/mpeg",
    b"\xff\xf3": "audio/mpeg",
    b"\xff\xf2": "audio/mpeg",
    b"fLaC": "audio/flac",
    b"OggS": "audio/ogg",
}


def _detect_audio_mime(audio_bytes: bytes) -> str:
    """根据魔数字节检测音频 MIME 类型，默认回退到 audio/wav。"""
    for magic, mime in _AUDIO_MIME_BY_MAGIC.items():
        if audio_bytes.startswith(magic):
            return mime
    return "audio/wav"


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

    audio_filename, audio_buffer = build_audio_file(request)
    del audio_filename
    audio_bytes = audio_buffer.getvalue()
    mime_type = _detect_audio_mime(audio_bytes)
    base64_str = base64.b64encode(audio_bytes).decode()
    data_uri = f"data:{mime_type};base64,{base64_str}"

    body = dict(transport.body)
    parameters = json_mapping_or_none(body.get("parameters"))
    if parameters is not None:
        parameters = dict(parameters)
    else:
        parameters = {}
    parameters.setdefault("result_format", "message")
    body["parameters"] = parameters
    body["input"] = {"messages": [{"role": "user", "content": [{"audio": data_uri}]}]}

    return body, transport.headers, transport.query


def parse_audio_transcription_response(
    payload: dict,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[str, dict | None]:
    """解析阿里云百炼 DashScope 多模态生成响应，提取转录文本。"""
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
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"]
        ]
        text = "\n".join(parts) if parts else None
    if text:
        raw_data = sanitize_json_object(payload) if options.include_raw_data else None
        return text, raw_data
    raise ValueError(build_parse_error_message(f"{DASHSCOPE_PROVIDER_LABEL} Audio Transcriptions", "响应缺少文本内容"))
