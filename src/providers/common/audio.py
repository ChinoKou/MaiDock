import base64
import binascii
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from ...i18n import runtime_actual, runtime_expected, runtime_item, runtime_subject, translate

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
        raise ValueError(translate("runtime.error.required", subject=provider_label, field="audio_base64"))
    try:
        encoded_size = len(audio_base64.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=f"{provider_label} audio_base64",
                expected=runtime_expected("valid_base64_data"),
                actual=runtime_actual("non_ascii_data"),
            )
        ) from exc
    if max_base64_chars is not None and encoded_size > max_base64_chars:
        raise ValueError(
            translate(
                "runtime.error.limit",
                subject=f"{provider_label} {runtime_subject('base64_data')}",
                limit=f"{max_base64_chars // (1024 * 1024)} MiB",
            )
        )
    if max_decoded_bytes is not None:
        max_encoded_size = ((max_decoded_bytes + 2) // 3) * 4
        if encoded_size > max_encoded_size:
            raise ValueError(
                translate(
                    "runtime.error.limit",
                    subject=f"{provider_label} {runtime_subject('audio_file')}",
                    limit=f"{max_decoded_bytes // (1024 * 1024)} MiB",
                )
            )
    try:
        decoded = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=f"{provider_label} audio_base64",
                expected=runtime_expected("valid_base64_data"),
                actual=runtime_actual("invalid_base64_data"),
            )
        ) from exc
    if not decoded:
        raise ValueError(
            translate("runtime.error.required", subject=provider_label, field=runtime_item("decoded_audio_data"))
        )
    if max_decoded_bytes is not None and len(decoded) > max_decoded_bytes:
        raise ValueError(
            translate(
                "runtime.error.limit",
                subject=f"{provider_label} {runtime_subject('audio_file')}",
                limit=f"{max_decoded_bytes // (1024 * 1024)} MiB",
            )
        )

    explicit_format = _explicit_audio_format(extra_params, provider_label=provider_label)
    detected_format = detect_audio_format(decoded)
    if explicit_format is not None and detected_format is not None and explicit_format != detected_format:
        raise ValueError(
            translate(
                "runtime.error.conflict_different",
                left=f"{provider_label} {runtime_subject('explicit_audio_format')} {explicit_format}",
                right=f"{runtime_subject('detected_file_signature')} {detected_format}",
            )
        )
    audio_format = explicit_format or detected_format
    if audio_format is None:
        raise ValueError(
            translate(
                "runtime.error.required",
                subject=f"{provider_label} {runtime_subject('unrecognized_audio')}",
                field=runtime_item("format_or_audio_format"),
            )
        )
    if audio_format not in allowed_formats:
        supported = ", ".join(sorted(allowed_formats))
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject=f"{provider_label} {runtime_subject('audio_format')} {audio_format}",
                allowed=supported,
            )
        )
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
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"{provider_label} {key}",
                    expected=runtime_expected("non_empty_string"),
                    actual=type(value).__name__,
                )
            )
        normalized = _AUDIO_FORMAT_ALIASES.get(value.strip().lower())
        if normalized is None:
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject=f"{provider_label} {runtime_subject('explicit_audio_format')}",
                    allowed=", ".join(sorted(_AUDIO_MIME_TYPES)),
                )
            )
        if resolved is not None and resolved != normalized:
            raise ValueError(
                translate(
                    "runtime.error.conflict_different",
                    left=f"{provider_label} {resolved_key}",
                    right=key,
                )
            )
        resolved = normalized
        resolved_key = key
    return resolved
