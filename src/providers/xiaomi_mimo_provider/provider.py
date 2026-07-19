import logging
from dataclasses import replace

import httpx
from maibot_sdk import LLMProviderBase

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
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    ResponseRequestSnapshot,
)
from ..chat_completions_family.transport import create_async_client, post_json
from .audio_transcriptions import build_mimo_audio_transcription
from .chat import (
    MIMO_CHAT_COMPLETIONS_ENDPOINT,
    build_chat_body,
    build_client_config,
    convert_response,
    resolve_path,
)
from .parameter_translation import mimo_thinking_enabled
from .reasoning import MimoReasoningManager
from .streaming import collect_mimo_stream_response

logger = logging.getLogger("maibot_plugin.maidock.xiaomi_mimo")


class XiaomiMimoProvider(LLMProviderBase):
    """小米 Mimo 原生 HTTP Provider。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
        state_store: PluginStateStore | None = None,
    ) -> None:
        self.options = options
        self._transport = transport
        if not options.mimo_force_disable_thinking and state_store is None:
            raise RuntimeError(translate("runtime.plugin.store_missing"))
        self._reasoning_manager = (
            MimoReasoningManager(
                state_store,
                retention_days=options.mimo_reasoning_retention_days,
            )
            if state_store is not None
            else None
        )

    async def get_response(self, request: dict) -> dict:
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
        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.mimo_user_agent,
            default_max_retries=self.options.mimo_max_retries,
            force_max_retries=self.options.mimo_force_max_retries,
            default_retry_interval=self.options.mimo_retry_interval,
            force_retry_interval=self.options.mimo_force_retry_interval,
        )
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

        path = resolve_path(config, MIMO_CHAT_COMPLETIONS_ENDPOINT)
        async with create_async_client(config, transport=self._transport) as client:
            if stream:
                result = await collect_mimo_stream_response(
                    client,
                    path,
                    body,
                    headers=extra_headers,
                    query=extra_query,
                    options=response_options,
                    max_retries=config.max_retries,
                    retry_interval=config.retry_interval,
                )
            else:
                payload = await post_json(
                    client,
                    path,
                    json_body=body,
                    headers=extra_headers,
                    query=extra_query,
                    provider_label="Xiaomi Mimo",
                    max_retries=config.max_retries,
                    retry_interval=config.retry_interval,
                )
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

    async def get_embedding(self, request: dict) -> dict:
        raise NotImplementedError(
            translate(
                "runtime.error.capability_unsupported",
                provider="Xiaomi Mimo Provider",
                capability="embedding",
            )
        )

    async def get_audio_transcriptions(self, request: dict) -> dict:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        result = await build_mimo_audio_transcription(
            request_model,
            options=self.options,
            transport=self._transport,
        )
        return result.to_host_dict()
