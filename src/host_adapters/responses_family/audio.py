from collections.abc import Collection, Mapping

from ...i18n import runtime_item, translate
from ...schemas import ProviderResponse
from ..common.audio import AudioFormat, prepare_base64_audio
from .responses import ResponsesMapper
from ...core.json_types import JsonValue


def build_responses_audio_input(
    audio_base64: str,
    format_hints: Mapping[str, object],
    *,
    prompt: str,
    provider_label: str,
    allowed_formats: Collection[AudioFormat],
    max_decoded_bytes: int | None = None,
    max_base64_chars: int | None = None,
) -> list[dict[str, JsonValue]]:
    """校验音频并构造 Responses input_audio/input_text 输入。"""

    audio = prepare_base64_audio(
        audio_base64,
        format_hints,
        provider_label=provider_label,
        allowed_formats=allowed_formats,
        max_decoded_bytes=max_decoded_bytes,
        max_base64_chars=max_base64_chars,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "audio_url": audio.data_url},
                {"type": "input_text", "text": prompt},
            ],
        }
    ]


def parse_responses_audio_transcription(
    payload: dict[str, JsonValue],
    *,
    mapper: ResponsesMapper,
    provider_label: str,
) -> ProviderResponse:
    """通过 Responses Mapper 解析音频转录结果。"""

    result = mapper.convert_response(payload)
    if not result.content:
        raise ValueError(
            translate(
                "runtime.error.response_missing",
                provider=provider_label,
                item=runtime_item("audio_transcription_text"),
            )
        )
    return result


__all__ = [
    "AudioFormat",
    "build_responses_audio_input",
    "parse_responses_audio_transcription",
]
