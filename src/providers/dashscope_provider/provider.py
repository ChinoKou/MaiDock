import logging

import httpx
from maibot_sdk import LLMProviderBase

from ...core.common import ProviderRuntimeOptions, log_request_summary, log_response_summary, read_model_identifier
from ...i18n import translate
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..common.httpx import HttpxClientConfig, create_async_client, post_json
from .audio_transcriptions import (
    build_audio_transcription_request,
    parse_audio_transcription_response,
)
from .chat import (
    DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT,
    DASHSCOPE_PROVIDER_LABEL,
    DashScopeEndpointKind,
    build_client_config,
    build_generation_body,
    convert_response,
    count_tools,
    dashscope_endpoint_path,
    is_dashscope_multimodal_model,
    is_multimodal_endpoint,
    is_dashscope_text_only_model,
    normalize_dashscope_model,
    request_has_image,
    resolve_path,
)
from .embeddings import build_dashscope_embedding_response, build_embedding_request
from .errors import DashScopeApiError, dashscope_error_factory
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
        self._endpoint_cache: dict[str, DashScopeEndpointKind] = {}

    def _resolve_chat_endpoint_kind(
        self,
        request: ResponseRequestSnapshot,
        model: str,
        *,
        has_image: bool,
    ) -> DashScopeEndpointKind:
        if has_image and is_dashscope_text_only_model(model):
            raise ValueError(translate("runtime.error.dashscope_visual_unsupported", model=model))
        if has_image:
            return "multimodal"

        normalized_model = normalize_dashscope_model(model)
        cached = self._endpoint_cache.get(normalized_model)
        if cached is not None:
            return cached
        if is_dashscope_text_only_model(model):
            return "text"
        if is_dashscope_multimodal_model(model):
            return "multimodal"
        if request.model_info.visual:
            return "multimodal"
        return "text"

    async def _request_chat(
        self,
        client: httpx.AsyncClient,
        path: str,
        body: dict,
        *,
        stream: bool,
        headers: dict[str, str],
        query: dict,
        config: HttpxClientConfig,
    ) -> ProviderResponse:
        if stream:
            return await collect_stream_response(
                client,
                path,
                body,
                headers={
                    **headers,
                    "Accept": "text/event-stream",
                    "X-Accel-Buffering": "no",
                    "X-DashScope-SSE": "enable",
                },
                query=query,
                options=self.options,
                max_retries=config.max_retries,
                retry_interval=config.retry_interval,
            )
        payload = await post_json(
            client,
            path,
            json_body=body,
            headers=headers,
            query=query,
            provider_label=DASHSCOPE_PROVIDER_LABEL,
            max_retries=config.max_retries,
            retry_interval=config.retry_interval,
            error_factory=dashscope_error_factory,
        )
        return convert_response(payload, options=self.options, is_multimodal=is_multimodal_endpoint(path))

    async def get_response(self, request: dict) -> dict:
        request_model = ResponseRequestSnapshot.model_validate(request)
        model = read_model_identifier(request_model.model_info)
        has_image = request_has_image(request_model.message_list)
        endpoint_kind = self._resolve_chat_endpoint_kind(request_model, model, has_image=has_image)
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
        path = dashscope_endpoint_path(config, endpoint_kind)
        async with create_async_client(config, transport=self._transport) as client:
            try:
                result = await self._request_chat(
                    client,
                    path,
                    body,
                    stream=stream,
                    headers=extra_headers,
                    query=extra_query,
                    config=config,
                )
            except DashScopeApiError as exc:
                if not (self.options.dashscope_auto_detect_endpoint and not has_image and exc.is_endpoint_mismatch):
                    raise
                alternate_kind: DashScopeEndpointKind = "multimodal" if endpoint_kind == "text" else "text"
                alternate_path = dashscope_endpoint_path(config, alternate_kind)
                logger.info(
                    translate(
                        "runtime.log.dashscope_endpoint_switch",
                        model=model,
                        from_endpoint=endpoint_kind,
                        to_endpoint=alternate_kind,
                    )
                )
                result = await self._request_chat(
                    client,
                    alternate_path,
                    body,
                    stream=stream,
                    headers=extra_headers,
                    query=extra_query,
                    config=config,
                )
                self._endpoint_cache[normalize_dashscope_model(model)] = alternate_kind

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
        endpoint, body, extra_headers, extra_query = build_embedding_request(request_model, options=self.options)
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
                error_factory=dashscope_error_factory,
            )
        return build_dashscope_embedding_response(payload, options=self.options).to_host_dict()

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
                error_factory=dashscope_error_factory,
            )

        content, raw_data = parse_audio_transcription_response(payload, options=self.options)
        return ProviderResponse(content=content, raw_data=raw_data).to_host_dict()
