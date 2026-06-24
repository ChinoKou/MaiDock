import logging

import httpx
from maibot_sdk import LLMProviderBase

from ...core.common import (
    ProviderRuntimeOptions,
    build_usage_from_snapshot,
    log_request_summary,
    log_response_summary,
)
from ...core.parameter_policy import apply_transport_parameter_policy
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ResponseRequestSnapshot,
)
from ...schemas.provider_contracts import ProviderResponse
from ..common.httpx import create_async_client, post_json, resolve_endpoint_path
from .embeddings import build_ark_embedding_response, build_embedding_request
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
    ) -> None:
        self.options = options
        self._transport = transport
        self._responses_mapper = create_responses_mapper(options=options, logger=logger)

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
        request_headers = transport.headers
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
        # VolcEngine 前缀缓存：检查是否需要创建/复用缓存
        prefix_cache_applied = False
        if self.options.volcengine_prefix_cache_enabled:
            from .prefix_cache import PrefixCacheManager
            cache_mgr = PrefixCacheManager.get_instance(
                cache_id_path=self.options.volcengine_prefix_cache_path
            )
            cache_params = await cache_mgr.resolve(
                model=upstream_request.model,
                messages=request_model.message_list,
            )
            if cache_params.get("caching"):
                # 需要创建前缀缓存 — 替换 body 中已有的 caching override
                body["caching"] = cache_params["caching"]
                body.pop("previous_response_id", None)
                body.pop("max_output_tokens", None)  # prefix cache 不支持该参数
                prefix_cache_applied = True
                logger.info(f"前缀缓存: 创建模式 model={upstream_request.model}")
            elif cache_params.get("previous_response_id"):
                # 复用已有缓存 — 替换 body，移除不能同时使用的参数
                body["previous_response_id"] = cache_params["previous_response_id"]
                body.pop("caching", None)
                body.pop("tools", None)  # 工具已在缓存中，不能重复设置
                logger.info(f"前缀缓存: 复用模式 model={upstream_request.model} id={cache_params['previous_response_id'][:24]}...")

        path = resolve_endpoint_path(
            config.base_url,
            api_prefix=VOLCENGINE_API_PREFIX,
            endpoint_path=ARK_RESPONSES_ENDPOINT,
        )
        async with create_async_client(config, transport=self._transport) as client:
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

        # 确认缓存创建（必须在 convert_response 之前，因为缓存创建响应无内容）
        if prefix_cache_applied and self.options.volcengine_prefix_cache_enabled:
            from .prefix_cache import PrefixCacheManager
            cache_mgr = PrefixCacheManager.get_instance(
                cache_id_path=self.options.volcengine_prefix_cache_path
            )
            response_id = payload.get("id") if isinstance(payload, dict) else None
            if response_id:
                expire_at = payload.get("expire_at") if isinstance(payload, dict) else None
                await cache_mgr.confirm(model=upstream_request.model, response_id=response_id, expire_at=expire_at)
            # 前缀缓存创建响应无 content/output，不走正常解析
            usage = build_usage_from_snapshot(payload.get("usage")) if isinstance(payload, dict) else None
            result = ProviderResponse(content=None, reasoning_content=None, tool_calls=[], usage=usage, raw_data=payload)
        else:
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
