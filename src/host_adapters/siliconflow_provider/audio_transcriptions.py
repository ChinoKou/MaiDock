import httpx

from ...core.common import RuntimeOptionsView
from ...schemas import AudioTranscriptionRequestSnapshot
from ..openai_auxiliary_family.audio_transcriptions import (
    build_multipart_audio_transcription_request,
    form_field_value,
    parse_multipart_audio_transcription_response,
)
from .parameter_translation import apply_siliconflow_audio_parameters
from ...core.json_types import JsonValue

SILICONFLOW_PROVIDER_LABEL = "siliconflow"


def _form_field_value(value: object) -> str:
    """保留原 Provider 内部入口，具体序列化由辅助协议 Family 负责。"""

    return form_field_value(value)


def build_audio_transcription_request(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: RuntimeOptionsView,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]], dict[str, str], dict[str, JsonValue]]:
    """通过辅助协议 Family 构建 SiliconFlow multipart 转录请求。"""

    return build_multipart_audio_transcription_request(
        request,
        options=options,
        provider_label=SILICONFLOW_PROVIDER_LABEL,
        policy_provider="siliconflow",
        apply_parameters=apply_siliconflow_audio_parameters,
    )


def parse_audio_transcription_response(
    response: httpx.Response,
    *,
    options: RuntimeOptionsView,
) -> tuple[str, dict[str, JsonValue] | None]:
    """解析 SiliconFlow 音频转录响应。"""

    return parse_multipart_audio_transcription_response(
        response,
        options=options,
        provider_label="SiliconFlow Audio Transcriptions",
    )
