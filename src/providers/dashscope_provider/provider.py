import logging

import httpx
from maibot_sdk import LLMProviderBase

from ...core.common import ProviderRuntimeOptions, log_request_summary, log_response_summary
from ...i18n import translate
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..common.httpx import HttpxClientConfig, HttpxProviderError, create_async_client, post_json
from .audio_transcriptions import (
    build_audio_transcription_request,
    parse_audio_transcription_response,
)
from .chat import (
    DASHSCOPE_GENERATION_ENDPOINT,
    DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT,
    DASHSCOPE_PROVIDER_LABEL,
    build_client_config,
    build_generation_body,
    convert_response,
    count_tools,
    is_multimodal_endpoint,
    resolve_path,
)
from .embeddings import build_dashscope_embedding_response, build_embedding_request
from .streaming import collect_stream_response

logger = logging.getLogger("maibot_plugin.maidock.dashscope")


class DashScopeProvider(LLMProviderBase):
    """阿里云百炼 DashScope 原生 HTTP Provider。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.options = options
        self._transport = transport
        self._endpoint_cache: dict[str, str] = {}

    def _resolve_chat_endpoint(self, config: HttpxClientConfig, model: str) -> str:
        cached = self._endpoint_cache.get(model)
        if cached is not None:
            return cached
        return resolve_path(config, DASHSCOPE_GENERATION_ENDPOINT)

    @staticmethod
    def _is_dashscope_endpoint_error(exc: HttpxProviderError) -> bool:
        message = str(exc)
        return "InvalidParameter" in message and "url" in message

    async def get_response(self, request: dict) -> dict:
        request_model = ResponseRequestSnapshot.model_validate(request)
        stream = bool(request_model.model_info.force_stream_mode)
        body, extra_headers, extra_query = build_generation_body(
            request_model, options=self.options, stream=stream, logger=logger
        )
        log_request_summary(
            logger,
            provider_label="dashscope",
            model=str(body["model"]),
            messages=len(request_model.message_list),
            tools=count_tools(body),
            extra=body,
            options=self.options,
        )

        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.dashscope_user_agent,
            force_official_endpoint=self.options.dashscope_force_official_endpoint,
            default_max_retries=self.options.dashscope_max_retries,
            force_max_retries=self.options.dashscope_force_max_retries,
            default_retry_interval=self.options.dashscope_retry_interval,
            force_retry_interval=self.options.dashscope_force_retry_interval,
        )
        model = str(body["model"])
        path = self._resolve_chat_endpoint(config, model)
        async with create_async_client(config, transport=self._transport) as client:
            try:
                if stream:
                    result = await collect_stream_response(
                        client,
                        path,
                        body,
                        headers={
                            **extra_headers,
                            "Accept": "text/event-stream",
                            "X-Accel-Buffering": "no",
                            "X-DashScope-SSE": "enable",
                        },
                        query=extra_query,
                        options=self.options,
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
                        provider_label=DASHSCOPE_PROVIDER_LABEL,
                        max_retries=config.max_retries,
                        retry_interval=config.retry_interval,
                    )
                    result = convert_response(payload, options=self.options, is_multimodal=is_multimodal_endpoint(path))
            except HttpxProviderError as exc:
                if not (
                    self.options.dashscope_auto_detect_endpoint
                    and self._is_dashscope_endpoint_error(exc)
                    and path == resolve_path(config, DASHSCOPE_GENERATION_ENDPOINT)
                ):
                    raise
                alt_path = resolve_path(config, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT)
                logger.info(translate("runtime.log.dashscope_endpoint_switch", model=model))
                if stream:
                    result = await collect_stream_response(
                        client,
                        alt_path,
                        body,
                        headers={
                            **extra_headers,
                            "Accept": "text/event-stream",
                            "X-Accel-Buffering": "no",
                            "X-DashScope-SSE": "enable",
                        },
                        query=extra_query,
                        options=self.options,
                        max_retries=config.max_retries,
                        retry_interval=config.retry_interval,
                    )
                else:
                    payload = await post_json(
                        client,
                        alt_path,
                        json_body=body,
                        headers=extra_headers,
                        query=extra_query,
                        provider_label=DASHSCOPE_PROVIDER_LABEL,
                        max_retries=config.max_retries,
                        retry_interval=config.retry_interval,
                    )
                    result = convert_response(
                        payload, options=self.options, is_multimodal=is_multimodal_endpoint(alt_path)
                    )
                self._endpoint_cache[model] = alt_path

        log_response_summary(
            logger,
            provider_label="dashscope",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    async def get_embedding(self, request: dict) -> dict:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        endpoint, body, extra_headers, extra_query, encoding_format = build_embedding_request(
            request_model, options=self.options
        )
        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.dashscope_user_agent,
            force_official_endpoint=self.options.dashscope_force_official_endpoint,
            default_max_retries=self.options.dashscope_max_retries,
            force_max_retries=self.options.dashscope_force_max_retries,
            default_retry_interval=self.options.dashscope_retry_interval,
            force_retry_interval=self.options.dashscope_force_retry_interval,
        )
        path = resolve_path(config, endpoint)
        async with create_async_client(config, transport=self._transport) as client:
            payload = await post_json(
                client,
                path,
                json_body=body,
                headers=extra_headers,
                query=extra_query,
                provider_label=f"{DASHSCOPE_PROVIDER_LABEL} Embeddings",
                max_retries=config.max_retries,
                retry_interval=config.retry_interval,
            )
        return build_dashscope_embedding_response(
            payload, options=self.options, encoding_format=encoding_format
        ).to_host_dict()

    async def get_audio_transcriptions(self, request: dict) -> dict:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query = build_audio_transcription_request(request_model, options=self.options)
        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.dashscope_user_agent,
            force_official_endpoint=self.options.dashscope_force_official_endpoint,
            default_max_retries=self.options.dashscope_max_retries,
            force_max_retries=self.options.dashscope_force_max_retries,
            default_retry_interval=self.options.dashscope_retry_interval,
            force_retry_interval=self.options.dashscope_force_retry_interval,
        )
        path = resolve_path(config, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT)
        async with create_async_client(config, transport=self._transport) as client:
            payload = await post_json(
                client,
                path,
                json_body=body,
                headers=extra_headers,
                query=extra_query,
                provider_label=f"{DASHSCOPE_PROVIDER_LABEL} Audio Transcriptions",
                max_retries=config.max_retries,
                retry_interval=config.retry_interval,
            )

        content, raw_data = parse_audio_transcription_response(payload, options=self.options)
        return ProviderResponse(content=content, raw_data=raw_data).to_host_dict()
