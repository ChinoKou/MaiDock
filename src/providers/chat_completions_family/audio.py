from collections.abc import Collection, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..common.audio import AudioFormat, PreparedAudio, prepare_base64_audio


class _AudioMessage(BaseModel):
    class TextPart(BaseModel):
        type: Literal["text"] = "text"
        text: str

    class InputData(BaseModel):
        data: str
        format: str | None = None

    class InputPart(BaseModel):
        type: Literal["input_audio"] = "input_audio"
        input_audio: "_AudioMessage.InputData"

    role: Literal["user"] = "user"
    content: list[Annotated[TextPart | InputPart, Field(discriminator="type")]]


def prepare_chat_audio(
    audio_base64: str,
    format_hints: Mapping[str, object],
    *,
    provider_label: str,
    allowed_formats: Collection[AudioFormat],
    max_decoded_bytes: int | None = None,
    max_base64_chars: int | None = None,
) -> PreparedAudio:
    """校验用于 Chat Completions content 的 Base64 音频。"""

    return prepare_base64_audio(
        audio_base64,
        format_hints,
        provider_label=provider_label,
        allowed_formats=allowed_formats,
        max_decoded_bytes=max_decoded_bytes,
        max_base64_chars=max_base64_chars,
    )


def build_chat_audio_message(
    audio: PreparedAudio,
    *,
    prompt: str | None = None,
    include_format: bool = False,
) -> dict:
    """构造标准 Chat Completions input_audio 用户消息。"""

    content: list[_AudioMessage.TextPart | _AudioMessage.InputPart] = [
        _AudioMessage.InputPart(
            input_audio=_AudioMessage.InputData(
                data=audio.data_url,
                format=audio.audio_format if include_format else None,
            )
        )
    ]
    if prompt is not None:
        content.append(_AudioMessage.TextPart(text=prompt))
    return _AudioMessage(content=content).model_dump(exclude_none=True)


__all__ = [
    "AudioFormat",
    "build_chat_audio_message",
    "prepare_chat_audio",
]
