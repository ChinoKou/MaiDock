import base64
import binascii
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, Field

from ...core.common import (
    ProviderRuntimeOptions,
    build_usage_from_snapshot,
    read_model_identifier,
)
from ...core.diagnostics import sanitize_json_object
from ...core.json_types import json_mapping_or_none, mapping_field
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    GenericUsageSnapshot,
    ProviderResponse,
)
from ..common.httpx import create_async_client, post_json
from .chat import (
    MIMO_CHAT_COMPLETIONS_ENDPOINT,
    build_client_config,
    resolve_path,
)

SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "mpeg"}
MIMO_AUDIO_TRANSCRIPTION_LABEL = "Xiaomi Mimo Audio Transcription"


class _AudioMessage(BaseModel):
    class TextPart(BaseModel):
        type: Literal["text"] = "text"
        text: str

    class InputData(BaseModel):
        data: str

    class InputPart(BaseModel):
        type: Literal["input_audio"] = "input_audio"
        input_audio: "_AudioMessage.InputData"

    role: Literal["user"] = "user"
    content: list[Annotated[TextPart | InputPart, Field(discriminator="type")]]


class _AudioTranscriptionBody(BaseModel):
    model: str
    messages: list[_AudioMessage]
    stream: bool = False


def _infer_audio_format(request: AudioTranscriptionRequestSnapshot) -> str | None:
    extra = request.extra_params.fields
    fmt = extra.get("format") or extra.get("audio_format")
    if isinstance(fmt, str) and fmt.strip().lower() in SUPPORTED_AUDIO_FORMATS:
        normalized = fmt.strip().lower()
        return "mpeg" if normalized == "mp3" else normalized
    return None


def _build_audio_data_url(audio_base64: str, audio_format: str | None) -> str:
    try:
        base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_base64 不是有效的 Base64 数据") from exc
    fmt = audio_format or "wav"
    return f"data:audio/{fmt};base64,{audio_base64}"


async def build_mimo_audio_transcription(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProviderResponse:
    if not request.audio_base64:
        raise ValueError("音频转写请求缺少 audio_base64")
    prompt: str = options.mimo_audio_transcription_prompt.strip()
    if not prompt:
        raise ValueError("未配置转录提示词，请在 Mimo 设置中填写 audio_transcription_prompt")
    audio_format: str | None = _infer_audio_format(request)
    data_url: str = _build_audio_data_url(request.audio_base64, audio_format)
    model: str = read_model_identifier(request.model_info)

    body = _AudioTranscriptionBody(
        model=model,
        messages=[
            _AudioMessage(
                content=[
                    _AudioMessage.TextPart(text=prompt),
                    _AudioMessage.InputPart(input_audio=_AudioMessage.InputData(data=data_url)),
                ],
            )
        ],
    ).model_dump()

    config = build_client_config(
        request.api_provider,
        user_agent=options.mimo_user_agent,
        default_max_retries=options.default_max_retries,
    )
    path = resolve_path(config, MIMO_CHAT_COMPLETIONS_ENDPOINT)

    async with create_async_client(config, transport=transport) as client:
        payload = await post_json(
            client,
            path,
            json_body=body,
            provider_label=MIMO_AUDIO_TRANSCRIPTION_LABEL,
            max_retries=options.default_max_retries,
        )

    return _parse_audio_transcription_response(payload, options=options)


def _parse_audio_transcription_response(payload: dict, *, options: ProviderRuntimeOptions) -> ProviderResponse:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{MIMO_AUDIO_TRANSCRIPTION_LABEL} 响应中没有 choices")
    first = json_mapping_or_none(choices[0])
    if first is None:
        raise ValueError(f"{MIMO_AUDIO_TRANSCRIPTION_LABEL} choices[0] 不是 object")
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
    if content is None:
        raise ValueError(f"{MIMO_AUDIO_TRANSCRIPTION_LABEL} 响应中无法提取转录文本")
    return ProviderResponse(
        content=content,
        usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
        raw_data=sanitize_json_object(payload) if options.include_raw_data else None,
    )
