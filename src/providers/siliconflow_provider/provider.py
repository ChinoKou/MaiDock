import logging

import httpx
from maibot_sdk import LLMProviderBase

from ...core.common import (
    ProviderRuntimeOptions,
    log_request_summary,
    log_response_summary,
)
from ...core.json_types import json_list_or_none
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..common.httpx import create_async_client, post_json, post_multipart
from .audio_transcriptions import build_audio_transcription_request, parse_audio_transcription_response
from .chat import (
    SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT,
    build_chat_body,
    build_client_config,
    convert_response,
    resolve_path,
)
from .embeddings import SILICONFLOW_EMBEDDINGS_ENDPOINT, build_embedding_request, build_siliconflow_embedding_response
from .streaming import collect_stream_response

logger = logging.getLogger("maibot_plugin.maidock.siliconflow")

SILICONFLOW_AUDIO_TRANSCRIPTIONS_ENDPOINT = "audio/transcriptions"


class SiliconFlowProvider(LLMProviderBase):
    """SiliconFlow 原生 HTTP Provider。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.options = options
        self._transport = transport

    async def get_response(self, request: dict) -> dict:
        request_model = ResponseRequestSnapshot.model_validate(request)
        stream = bool(request_model.model_info.force_stream_mode)
        body, extra_headers, extra_query = build_chat_body(
            request_model,
            options=self.options,
            logger=logger,
            stream=stream,
        )
        tool_count = len(json_list_or_none(body.get("tools")) or [])
        message_count = len(json_list_or_none(body.get("messages")) or [])
        log_request_summary(
            logger,
            provider_label="siliconflow",
            model=str(body["model"]),
            messages=message_count,
            tools=tool_count,
            extra=body,
            options=self.options,
        )

        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.siliconflow_user_agent,
            force_official_endpoint=self.options.siliconflow_force_official_endpoint,
        )
        path = resolve_path(config, SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT)
        async with create_async_client(config, transport=self._transport) as client:
            if stream:
                result = await collect_stream_response(
                    client,
                    path,
                    body,
                    headers=extra_headers,
                    query=extra_query,
                    options=self.options,
                    max_retries=self.options.default_max_retries,
                )
            else:
                payload = await post_json(
                    client,
                    path,
                    json_body=body,
                    headers=extra_headers,
                    query=extra_query,
                    provider_label="SiliconFlow",
                )
                result = convert_response(payload, options=self.options)

        log_response_summary(
            logger,
            provider_label="siliconflow",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    async def get_embedding(self, request: dict) -> dict:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query, encoding_format = build_embedding_request(request_model, options=self.options)
        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.siliconflow_user_agent,
            force_official_endpoint=self.options.siliconflow_force_official_endpoint,
        )
        path = resolve_path(config, SILICONFLOW_EMBEDDINGS_ENDPOINT)
        async with create_async_client(config, transport=self._transport) as client:
            payload = await post_json(
                client,
                path,
                json_body=body,
                headers=extra_headers,
                query=extra_query,
                provider_label="SiliconFlow Embeddings",
            )

        return build_siliconflow_embedding_response(
            payload, options=self.options, encoding_format=encoding_format
        ).to_host_dict()

    async def get_audio_transcriptions(self, request: dict) -> dict:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        form_data, files, extra_headers, extra_query = build_audio_transcription_request(
            request_model, options=self.options
        )

        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.siliconflow_user_agent,
            force_official_endpoint=self.options.siliconflow_force_official_endpoint,
        )
        path = resolve_path(config, SILICONFLOW_AUDIO_TRANSCRIPTIONS_ENDPOINT)
        async with create_async_client(config, transport=self._transport) as client:
            response = await post_multipart(
                client,
                path,
                form_data=form_data,
                files=files,
                headers=extra_headers,
                query=extra_query,
                provider_label="SiliconFlow Audio Transcriptions",
            )

        content, raw_data = parse_audio_transcription_response(response, options=self.options)
        return ProviderResponse(content=content, raw_data=raw_data).to_host_dict()
