import base64
import binascii
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

type AudioFormat = Literal["wav", "mp3", "aac", "m4a", "flac", "ogg"]

_AUDIO_MIME_TYPES: dict[AudioFormat, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "m4a": "audio/m4a",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}
_AUDIO_FORMAT_ALIASES: dict[str, AudioFormat] = {
    "wave": "wav",
    "wav": "wav",
    "mpeg": "mp3",
    "mp3": "mp3",
    "aac": "aac",
    "m4a": "m4a",
    "flac": "flac",
    "ogg": "ogg",
}


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """已校验并确定格式的 Base64 音频。"""

    audio_format: AudioFormat
    mime_type: str
    decoded_bytes: bytes
    base64_data: str

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64_data}"


def prepare_base64_audio(
    audio_base64: str,
    extra_params: Mapping[str, object],
    *,
    provider_label: str,
    allowed_formats: Collection[AudioFormat],
    max_decoded_bytes: int | None = None,
    max_base64_chars: int | None = None,
) -> PreparedAudio:
    """严格解码 Base64，并结合显式提示与文件签名确定音频格式。"""

    if not audio_base64:
        raise ValueError(f"{provider_label} 请求缺少 audio_base64")
    try:
        encoded_size = len(audio_base64.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{provider_label} audio_base64 不是有效的 Base64 数据") from exc
    if max_base64_chars is not None and encoded_size > max_base64_chars:
        raise ValueError(f"{provider_label} Base64 字符串超过 {max_base64_chars // (1024 * 1024)} MiB 限制")
    if max_decoded_bytes is not None:
        max_encoded_size = ((max_decoded_bytes + 2) // 3) * 4
        if encoded_size > max_encoded_size:
            raise ValueError(f"{provider_label} 音频文件超过 {max_decoded_bytes // (1024 * 1024)} MiB 限制")
    try:
        decoded = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{provider_label} audio_base64 不是有效的 Base64 数据") from exc
    if not decoded:
        raise ValueError(f"{provider_label} audio_base64 解码后为空")
    if max_decoded_bytes is not None and len(decoded) > max_decoded_bytes:
        raise ValueError(f"{provider_label} 音频文件超过 {max_decoded_bytes // (1024 * 1024)} MiB 限制")

    explicit_format = _explicit_audio_format(extra_params, provider_label=provider_label)
    detected_format = detect_audio_format(decoded)
    if explicit_format is not None and detected_format is not None and explicit_format != detected_format:
        raise ValueError(f"{provider_label} 显式音频格式 {explicit_format} 与文件签名 {detected_format} 不一致")
    audio_format = explicit_format or detected_format
    if audio_format is None:
        raise ValueError(f"{provider_label} 无法识别音频格式，请通过 format 或 audio_format 明确指定")
    if audio_format not in allowed_formats:
        supported = ", ".join(sorted(allowed_formats))
        raise ValueError(f"{provider_label} 不支持 {audio_format} 音频，仅支持: {supported}")
    return PreparedAudio(
        audio_format=audio_format,
        mime_type=_AUDIO_MIME_TYPES[audio_format],
        decoded_bytes=decoded,
        base64_data=audio_base64,
    )


def detect_audio_format(data: bytes) -> AudioFormat | None:
    """通过常见容器和帧头签名识别音频格式。"""

    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"OggS"):
        return "ogg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "m4a"
    if len(data) >= 2 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0:
        return "aac"
    if data.startswith(b"ID3"):
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0 and data[1] & 0x06:
        return "mp3"
    return None


def _explicit_audio_format(
    extra_params: Mapping[str, object],
    *,
    provider_label: str,
) -> AudioFormat | None:
    resolved: AudioFormat | None = None
    resolved_key = ""
    for key in ("format", "audio_format"):
        value = extra_params.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{provider_label} {key} 必须是非空字符串")
        normalized = _AUDIO_FORMAT_ALIASES.get(value.strip().lower())
        if normalized is None:
            raise ValueError(f"{provider_label} 不支持的显式音频格式: {value}")
        if resolved is not None and resolved != normalized:
            raise ValueError(f"{provider_label} {resolved_key} 与 {key} 指定了不同音频格式")
        resolved = normalized
        resolved_key = key
    return resolved
