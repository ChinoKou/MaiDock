from dataclasses import replace

import logging

from ...clients.mimo import MimoClient, MimoConnection
from ...core.common import (
    ProviderRuntimeOptions,
    log_request_summary,
    log_response_summary,
    read_api_key,
    read_model_identifier,
)
from ...core.json_types import json_list_or_none
from ...core.state_store import PluginStateStore
from ...i18n import runtime_item, runtime_subject, translate
from ...schemas import AudioTranscriptionRequestSnapshot, ResponseRequestSnapshot
from ..common.client_bridge import build_http_connection, build_retry_policy, json_resource_request
from ..chat_completions_family.transport import HttpxClientConfig
from ..common.options import build_host_common_options, build_mimo_host_options
from ..common.rpc import HostRpcRequest, HostRpcResponse
from .audio_transcriptions import build_mimo_audio_transcription_request, parse_mimo_audio_transcription_response
from .chat import MIMO_CHAT_COMPLETIONS_ENDPOINT, build_chat_body, build_client_config, convert_response, resolve_path
from .parameter_translation import mimo_thinking_enabled
from .reasoning import MimoReasoningManager
from .streaming import collect_mimo_stream_response

logger = logging.getLogger("maibot_plugin.maidock.xiaomi_mimo")


class MimoHostAdapter:
    """MaiBot Host 合约到 Xiaomi Mimo 原生资源的适配器。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        client: MimoClient,
        state_store: PluginStateStore | None = None,
    ) -> None:
        self.options = build_host_common_options(options)
        self.vendor_options = build_mimo_host_options(options)
        self.client = client
        self._reasoning_manager = (
            MimoReasoningManager(state_store, retention_days=self.vendor_options.reasoning_retention_days)
            if state_store is not None
            else None
        )

    def _connection(
        self, request: ResponseRequestSnapshot | AudioTranscriptionRequestSnapshot
    ) -> tuple[MimoConnection, HttpxClientConfig]:
        connection_options = self.vendor_options.connection
        config = build_client_config(
            request.api_provider,
            user_agent=connection_options.user_agent,
            default_max_retries=connection_options.max_retries,
            force_max_retries=connection_options.force_max_retries,
            default_retry_interval=connection_options.retry_interval,
            force_retry_interval=connection_options.force_retry_interval,
        )
        return (
            MimoConnection(
                http=build_http_connection(config),
                retry=build_retry_policy(config),
                chat_completions_path=resolve_path(config, MIMO_CHAT_COMPLETIONS_ENDPOINT),
            ),
            config,
        )

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = ResponseRequestSnapshot.model_validate(request)
        stream = bool(request_model.model_info.force_stream_mode)
        body, extra_headers, extra_query = build_chat_body(
            request_model,
            options=self.options,
            logger=logger,
            stream=stream,
        )
        model = read_model_identifier(request_model.model_info)
        api_key = read_api_key(request_model.api_provider)
        connection, config = self._connection(request_model)
        thinking_enabled = mimo_thinking_enabled(body)
        hide_reasoning = thinking_enabled and self.options.reasoning_parse_mode == "none"
        response_options = replace(self.options, reasoning_parse_mode="native") if hide_reasoning else self.options
        if thinking_enabled:
            if self._reasoning_manager is None:
                raise RuntimeError(
                    translate(
                        "runtime.error.required",
                        subject=runtime_subject("mimo_native_reasoning"),
                        field=runtime_item("reasoning_manager"),
                    )
                )
            await self._reasoning_manager.restore_history(
                request_model,
                body,
                base_url=config.base_url,
                api_key=api_key,
                model=model,
            )
        log_request_summary(
            logger,
            provider_label="xiaomi-mimo",
            model=str(body["model"]),
            messages=len(json_list_or_none(body.get("messages")) or []),
            tools=len(json_list_or_none(body.get("tools")) or []),
            extra=body,
            options=self.options,
        )
        resource_request = json_resource_request(body, headers=extra_headers, query=extra_query)
        async with self.client.session(connection) as session:
            if stream:
                result = await collect_mimo_stream_response(
                    session.chat_completions.stream(resource_request, retry=session.retry),
                    options=response_options,
                )
            else:
                payload = await session.chat_completions.create(resource_request, retry=session.retry)
                result = convert_response(payload, options=response_options)
        if self._reasoning_manager is not None:
            await self._reasoning_manager.preserve_response(
                result,
                base_url=config.base_url,
                api_key=api_key,
                model=model,
                thinking_enabled=thinking_enabled,
            )
        if hide_reasoning:
            result.reasoning_content = None
        log_response_summary(
            logger,
            provider_label="xiaomi-mimo",
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
                provider="Xiaomi Mimo Provider",
                capability="embedding",
            )
        )

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query = build_mimo_audio_transcription_request(
            request_model,
            options=self.options,
        )
        connection, _ = self._connection(request_model)
        async with self.client.session(connection) as session:
            payload = await session.chat_completions.create(
                json_resource_request(body, headers=extra_headers, query=extra_query),
                retry=session.retry,
            )
        return parse_mimo_audio_transcription_response(payload, options=self.options).to_host_dict()
