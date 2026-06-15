import logging

import httpx
from maibot_sdk import LLMProviderBase

from ...core.common import (
    ProviderRuntimeOptions,
    log_request_summary,
    log_response_summary,
)
from ...schemas import ResponseRequestSnapshot
from ..common.httpx import create_async_client, post_json, resolve_endpoint_path
from .messages import (
    ANTHROPIC_API_PREFIX,
    ANTHROPIC_MESSAGES_ENDPOINT,
    ANTHROPIC_PROVIDER_LABEL,
    build_client_config,
    build_http_body,
    build_request,
    convert_response,
)
from .streaming import collect_stream_response

logger = logging.getLogger("maibot_plugin.maidock.anthropic_messages")


class AnthropicMessagesProvider(LLMProviderBase):
    """基于 Anthropic Messages API 的原生 HTTP Provider。"""

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
        upstream_request = build_request(request_model, options=self.options, logger=logger)
        stream = bool(request_model.model_info.force_stream_mode)
        body = build_http_body(upstream_request, options=self.options, stream=stream)
        log_request_summary(
            logger,
            provider_label="anthropic",
            model=upstream_request.model,
            messages=len(upstream_request.messages),
            tools=len(upstream_request.tools),
            extra=body,
            options=self.options,
        )

        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.anthropic_user_agent,
            default_max_retries=self.options.anthropic_max_retries,
            force_max_retries=self.options.anthropic_force_max_retries,
            default_retry_interval=self.options.anthropic_retry_interval,
            force_retry_interval=self.options.anthropic_force_retry_interval,
        )
        path = resolve_endpoint_path(
            config.base_url,
            api_prefix=ANTHROPIC_API_PREFIX,
            endpoint_path=ANTHROPIC_MESSAGES_ENDPOINT,
        )
        async with create_async_client(config, transport=self._transport) as client:
            if stream:
                stream_headers = dict(upstream_request.extra_headers)
                stream_headers.setdefault("Accept", "text/event-stream")
                payload = await collect_stream_response(
                    client,
                    path,
                    body,
                    headers=stream_headers,
                    query=upstream_request.extra_query,
                    parse_mode=self.options.tool_argument_parse_mode,
                    max_retries=config.max_retries,
                    retry_interval=config.retry_interval,
                )
            else:
                payload = await post_json(
                    client,
                    path,
                    json_body=body,
                    headers=upstream_request.extra_headers,
                    query=upstream_request.extra_query,
                    provider_label=ANTHROPIC_PROVIDER_LABEL,
                    max_retries=config.max_retries,
                    retry_interval=config.retry_interval,
                )

        result = convert_response(payload, options=self.options)
        log_response_summary(
            logger,
            provider_label="anthropic",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    async def get_embedding(self, request: dict) -> dict:
        del request
        raise NotImplementedError("Anthropic Messages API 不提供 embedding 端点，请改用支持 embedding 的 Provider")

    async def get_audio_transcriptions(self, request: dict) -> dict:
        del request
        raise NotImplementedError(
            "Anthropic Messages API 不提供 audio_transcription 端点，请改用支持音频转写的 Provider"
        )
