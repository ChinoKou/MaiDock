import httpx

from ...core.common import ProviderRuntimeOptions
from ...schemas import AudioTranscriptionRequestSnapshot
from ..openai_auxiliary_family.audio_transcriptions import (
    build_multipart_audio_transcription_request,
    parse_multipart_audio_transcription_response,
)
from .parameter_translation import apply_openai_audio_parameters
from .responses import OPENAI_PROVIDER_LABEL


def build_audio_transcription_request(
    request: AudioTranscriptionRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]], dict[str, str], dict]:
    return build_multipart_audio_transcription_request(
        request,
        options=options,
        provider_label=OPENAI_PROVIDER_LABEL,
        policy_provider="openai_responses",
        apply_parameters=apply_openai_audio_parameters,
    )


def parse_audio_transcription_response(
    response: httpx.Response,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[str, dict | None]:
    return parse_multipart_audio_transcription_response(
        response,
        options=options,
        provider_label="OpenAI Audio Transcriptions",
    )
