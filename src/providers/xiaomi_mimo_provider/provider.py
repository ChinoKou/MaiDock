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
    ResponseRequestSnapshot,
)
from ..common.httpx import create_async_client, post_json
from .audio_transcriptions import build_mimo_audio_transcription
from .chat import (
    MIMO_CHAT_COMPLETIONS_ENDPOINT,
    build_chat_body,
    build_client_config,
    convert_response,
    resolve_path,
)
from .streaming import collect_mimo_stream_response

logger = logging.getLogger("maibot_plugin.maidock.xiaomi_mimo")


class XiaomiMimoProvider(LLMProviderBase):
    """小米 Mimo 原生 HTTP Provider。"""

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
            provider_label="xiaomi-mimo",
            model=str(body["model"]),
            messages=message_count,
            tools=tool_count,
            extra=body,
            options=self.options,
        )

        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.mimo_user_agent,
            default_max_retries=self.options.default_max_retries,
        )
        path = resolve_path(config, MIMO_CHAT_COMPLETIONS_ENDPOINT)
        async with create_async_client(config, transport=self._transport) as client:
            if stream:
                result = await collect_mimo_stream_response(
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
                    provider_label="Xiaomi Mimo",
                    max_retries=self.options.default_max_retries,
                )
                result = convert_response(payload, options=self.options)

        log_response_summary(
            logger,
            provider_label="xiaomi-mimo",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    async def get_embedding(self, request: dict) -> dict:
        raise NotImplementedError("Xiaomi Mimo Provider 当前不提供 embedding 端点")

    async def get_audio_transcriptions(self, request: dict) -> dict:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        result = await build_mimo_audio_transcription(
            request_model,
            options=self.options,
            transport=self._transport,
        )
        return result.to_host_dict()
