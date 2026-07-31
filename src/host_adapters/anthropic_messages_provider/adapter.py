import logging

from ...clients.anthropic import AnthropicClient, AnthropicConnection
from ...core.common import ProviderRuntimeOptions, log_request_summary, log_response_summary
from ...i18n import translate
from ...schemas import ResponseRequestSnapshot
from ..common.client_bridge import build_http_connection, build_retry_policy, json_resource_request
from ..common.httpx import resolve_endpoint_path
from ..common.options import build_anthropic_host_options, build_host_common_options
from ..common.rpc import HostRpcRequest, HostRpcResponse
from .messages import (
    ANTHROPIC_API_PREFIX,
    ANTHROPIC_MESSAGES_ENDPOINT,
    build_client_config,
    build_http_body,
    build_request,
    convert_response,
)
from .streaming import collect_stream_response

logger = logging.getLogger("maibot_plugin.maidock.anthropic_messages")


class AnthropicHostAdapter:
    """MaiBot Host 合约到 Anthropic Messages 的适配器。"""

    def __init__(self, *, options: ProviderRuntimeOptions, client: AnthropicClient) -> None:
        self.options = build_host_common_options(options)
        self.vendor_options = build_anthropic_host_options(options)
        self.client = client

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
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
        connection_options = self.vendor_options.connection
        config = build_client_config(
            request_model.api_provider,
            user_agent=connection_options.user_agent,
            default_max_retries=connection_options.max_retries,
            force_max_retries=connection_options.force_max_retries,
            default_retry_interval=connection_options.retry_interval,
            force_retry_interval=connection_options.force_retry_interval,
        )
        connection = AnthropicConnection(
            http=build_http_connection(config),
            retry=build_retry_policy(config),
            messages_path=resolve_endpoint_path(
                config.base_url,
                api_prefix=ANTHROPIC_API_PREFIX,
                endpoint_path=ANTHROPIC_MESSAGES_ENDPOINT,
            ),
        )
        headers = dict(upstream_request.extra_headers)
        if stream:
            headers.setdefault("Accept", "text/event-stream")
        resource_request = json_resource_request(body, headers=headers, query=upstream_request.extra_query)
        async with self.client.session(connection) as session:
            if stream:
                payload = await collect_stream_response(
                    session.messages.stream(resource_request, retry=session.retry),
                    parse_mode=self.options.tool_argument_parse_mode,
                )
            else:
                payload = await session.messages.create(resource_request, retry=session.retry)
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

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
        del request
        raise NotImplementedError(
            translate(
                "runtime.error.capability_unsupported",
                provider="Anthropic Messages API",
                capability="embedding",
            )
        )

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        del request
        raise NotImplementedError(
            translate(
                "runtime.error.capability_unsupported",
                provider="Anthropic Messages API",
                capability="audio_transcription",
            )
        )
