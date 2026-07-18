from maibot_sdk import LLMProviderBase
import httpx
import logging

from ...core.common import (
    ProviderRuntimeOptions,
    log_request_summary,
    log_response_summary,
)
from ...core.parameter_policy import apply_transport_parameter_policy
from ...core.state_store import PluginStateStore
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ResponseRequestSnapshot,
)
from ..common.httpx import (
    HttpxProviderError,
    create_async_client,
    post_json,
    resolve_endpoint_path,
)
from .embeddings import build_ark_embedding_response, build_embedding_request
from .prefix_cache import (
    ARK_TOKENIZATION_ENDPOINT,
    PrefixCacheManager,
    PrefixCacheResolution,
)
from .responses import (
    ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT,
    ARK_RESPONSES_ENDPOINT,
    VOLCENGINE_API_PREFIX,
    VOLCENGINE_PROVIDER_LABEL,
    build_ark_request_headers,
    build_client_config,
    create_responses_mapper,
)
from .streaming import collect_ark_response_stream

logger = logging.getLogger("maibot_plugin.maidock.volcengine_ark")


class VolcengineArkResponsesProvider(LLMProviderBase):
    """Volcengine Ark 原生 HTTP Provider，兼容 Responses API。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
        state_store: PluginStateStore | None = None,
    ) -> None:
        self.options = options
        self._transport = transport
        self._responses_mapper = create_responses_mapper(options=options, logger=logger)
        if options.volcengine_prefix_cache_enabled and state_store is None:
            raise RuntimeError("ARK 前缀缓存已启用，但未注入 MaiDock 持久化存储")
        self._prefix_cache_manager = (
            PrefixCacheManager(
                state_store,
                ttl_seconds=options.volcengine_prefix_cache_ttl_seconds,
            )
            if state_store is not None and options.volcengine_prefix_cache_enabled
            else None
        )

    async def get_response(self, request: dict) -> dict:
        request_model = ResponseRequestSnapshot.model_validate(request)
        upstream_request = self._responses_mapper.build_request(request_model)
        stream = bool(request_model.model_info.force_stream_mode)
        body = self._responses_mapper.build_http_body(upstream_request, stream=stream, apply_policy=False)
        policy = self.options.parameter_policies.get("volcengine_ark", "response")
        request_headers = build_ark_request_headers(upstream_request.extra_headers, body)
        transport = apply_transport_parameter_policy(
            body=body,
            headers=request_headers,
            query=upstream_request.extra_query,
            policy=policy,
            provider_label=VOLCENGINE_PROVIDER_LABEL,
            capability="response",
        )
        body = transport.body
        request_headers = build_ark_request_headers(transport.headers, body)
        upstream_request.extra_query.clear()
        upstream_request.extra_query.update(transport.query)
        log_request_summary(
            logger,
            provider_label="volcengine-ark-responses",
            model=upstream_request.model,
            messages=len(upstream_request.input),
            tools=len(upstream_request.tool_params()),
            extra=body,
            options=self.options,
        )

        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.volcengine_user_agent,
            force_official_endpoint=self.options.volcengine_force_official_endpoint,
            default_max_retries=self.options.volcengine_max_retries,
            force_max_retries=self.options.volcengine_force_max_retries,
            default_retry_interval=self.options.volcengine_retry_interval,
            force_retry_interval=self.options.volcengine_force_retry_interval,
        )
        path = resolve_endpoint_path(
            config.base_url,
            api_prefix=VOLCENGINE_API_PREFIX,
            endpoint_path=ARK_RESPONSES_ENDPOINT,
        )
        tokenization_path = resolve_endpoint_path(
            config.base_url,
            api_prefix=VOLCENGINE_API_PREFIX,
            endpoint_path=ARK_TOKENIZATION_ENDPOINT,
        )
        resolution = PrefixCacheResolution(body=body)
        async with create_async_client(config, transport=self._transport) as client:
            if self._prefix_cache_manager is not None:
                resolution = await self._prefix_cache_manager.resolve(
                    client,
                    responses_path=path,
                    tokenization_path=tokenization_path,
                    body=body,
                    headers=request_headers,
                    query=upstream_request.extra_query,
                    client_config=config,
                    max_retries=config.max_retries,
                    retry_interval=config.retry_interval,
                )
                body = resolution.body
            try:
                if stream:
                    payload = await collect_ark_response_stream(
                        client,
                        path,
                        body,
                        headers=request_headers,
                        query=upstream_request.extra_query,
                        model=upstream_request.model,
                        max_retries=config.max_retries,
                        retry_interval=config.retry_interval,
                    )
                else:
                    payload = await post_json(
                        client,
                        path,
                        json_body=body,
                        headers=request_headers,
                        query=upstream_request.extra_query,
                        provider_label=VOLCENGINE_PROVIDER_LABEL,
                        max_retries=config.max_retries,
                        retry_interval=config.retry_interval,
                    )
            except HttpxProviderError as exc:
                if (
                    resolution.cache_key is not None
                    and exc.status_code is not None
                    and 400 <= exc.status_code < 500
                    and self._prefix_cache_manager is not None
                ):
                    await self._prefix_cache_manager.invalidate(resolution.cache_key)
                raise

        result = self._responses_mapper.convert_response(payload)
        log_response_summary(
            logger,
            provider_label="volcengine-ark-responses",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    async def get_embedding(self, request: dict) -> dict:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query, encoding_format = build_embedding_request(request_model, options=self.options)
        request_headers = build_ark_request_headers(extra_headers, body)
        policy = self.options.parameter_policies.get("volcengine_ark", "embeddings")
        transport = apply_transport_parameter_policy(
            body=body,
            headers=request_headers,
            query=extra_query,
            policy=policy,
            provider_label=VOLCENGINE_PROVIDER_LABEL,
            capability="embeddings",
        )
        body = transport.body
        request_headers = transport.headers
        extra_query = transport.query
        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.volcengine_user_agent,
            force_official_endpoint=self.options.volcengine_force_official_endpoint,
            default_max_retries=self.options.volcengine_max_retries,
            force_max_retries=self.options.volcengine_force_max_retries,
            default_retry_interval=self.options.volcengine_retry_interval,
            force_retry_interval=self.options.volcengine_force_retry_interval,
        )
        path = resolve_endpoint_path(
            config.base_url,
            api_prefix=VOLCENGINE_API_PREFIX,
            endpoint_path=ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT,
        )
        async with create_async_client(config, transport=self._transport) as client:
            payload = await post_json(
                client,
                path,
                json_body=body,
                headers=request_headers,
                query=extra_query,
                provider_label="Volcengine Ark Embeddings",
                max_retries=config.max_retries,
                retry_interval=config.retry_interval,
            )

        return build_ark_embedding_response(
            payload, options=self.options, encoding_format=encoding_format
        ).to_host_dict()

    async def get_audio_transcriptions(self, request: dict) -> dict:
        AudioTranscriptionRequestSnapshot.model_validate(request)
        raise NotImplementedError("Volcengine Ark Provider 当前不提供 audio_transcription 端点")
