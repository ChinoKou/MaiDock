import logging

from ...clients.ark import ArkClient, ArkConnection
from ...clients.common import ClientHttpError
from ...core.common import ProviderRuntimeOptions, log_request_summary, log_response_summary
from ...core.state_store import PluginStateStore
from ...i18n import translate
from ...schemas import AudioTranscriptionRequestSnapshot, EmbeddingRequestSnapshot, ResponseRequestSnapshot
from ..common.client_bridge import build_http_connection, build_retry_policy, json_resource_request
from ..responses_family.transport import HttpxClientConfig, resolve_endpoint_path
from ..common.options import build_ark_host_options, build_host_common_options
from ..common.rpc import HostRpcRequest, HostRpcResponse
from .audio_transcriptions import build_ark_audio_transcription_request, parse_ark_audio_transcription_response
from .embeddings import build_ark_embedding_response, build_embedding_request
from .prefix_cache import ARK_TOKENIZATION_ENDPOINT, PrefixCacheManager, PrefixCacheResolution
from .responses import (
    ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT,
    ARK_RESPONSES_ENDPOINT,
    VOLCENGINE_API_PREFIX,
    build_ark_request_headers,
    build_client_config,
    builtin_endpoint_profile,
    create_responses_mapper,
)
from .streaming import collect_ark_response_stream

logger = logging.getLogger("maibot_plugin.maidock.volcengine_ark")


class ArkHostAdapter:
    """MaiBot Host 合约到 Volcengine ARK 原生资源的适配器。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        client: ArkClient,
        state_store: PluginStateStore | None = None,
    ) -> None:
        self.options = build_host_common_options(options)
        self.vendor_options = build_ark_host_options(options)
        self.client = client
        self._responses_mapper = create_responses_mapper(options=self.options, logger=logger)
        # 订阅制内置端点（Agent Plan / Coding Plan）生效时停用前缀缓存：
        # tokenization 与 caching/previous_response_id 只在按量付费的 /api/v3 有文档，
        # 订阅端点上这些辅助调用会把 404 搅进主请求链路。缓存被停用时也不再要求
        # state_store——一个不会运行的功能不该阻断适配器构造。
        plan_endpoint_active = (
            self.vendor_options.force_official_endpoint and self.vendor_options.builtin_endpoint_mode != "standard"
        )
        if self.vendor_options.prefix_cache_enabled and plan_endpoint_active:
            logger.info(translate("runtime.log.cache_disabled_plan_endpoint"))
        prefix_cache_active = self.vendor_options.prefix_cache_enabled and not plan_endpoint_active
        if prefix_cache_active and state_store is None:
            raise RuntimeError(translate("runtime.plugin.cache_store_missing"))
        self._prefix_cache_manager = (
            PrefixCacheManager(state_store, ttl_seconds=self.vendor_options.prefix_cache_ttl_seconds)
            if state_store is not None and prefix_cache_active
            else None
        )

    def _connection(
        self, request: ResponseRequestSnapshot | EmbeddingRequestSnapshot | AudioTranscriptionRequestSnapshot
    ) -> tuple[ArkConnection, HttpxClientConfig]:
        connection_options = self.vendor_options.connection
        config = build_client_config(
            request.api_provider,
            user_agent=connection_options.user_agent,
            force_official_endpoint=self.vendor_options.force_official_endpoint,
            builtin_endpoint_mode=self.vendor_options.builtin_endpoint_mode,
            default_max_retries=connection_options.max_retries,
            force_max_retries=connection_options.force_max_retries,
            default_retry_interval=connection_options.retry_interval,
            force_retry_interval=connection_options.force_retry_interval,
        )
        # 前缀必须与实际生效的 base 匹配：只有强制原生 endpoint 时订阅端点的
        # plan 前缀才会出现在 base 里；Host base_url 场景维持既有 api/v3 语义。
        if self.vendor_options.force_official_endpoint:
            _, api_prefix = builtin_endpoint_profile(self.vendor_options.builtin_endpoint_mode)
        else:
            api_prefix = VOLCENGINE_API_PREFIX

        def path(endpoint: str) -> str:
            return resolve_endpoint_path(config.base_url, api_prefix=api_prefix, endpoint_path=endpoint)

        return (
            ArkConnection(
                http=build_http_connection(config),
                retry=build_retry_policy(config),
                responses_path=path(ARK_RESPONSES_ENDPOINT),
                embeddings_path=path(ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT),
                audio_transcriptions_path=path(ARK_RESPONSES_ENDPOINT),
                tokenization_path=path(ARK_TOKENIZATION_ENDPOINT),
            ),
            config,
        )

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = ResponseRequestSnapshot.model_validate(request)
        upstream_request = self._responses_mapper.build_request(request_model)
        stream = bool(request_model.model_info.force_stream_mode)
        body = self._responses_mapper.build_http_body(upstream_request, stream=stream)
        request_headers = build_ark_request_headers(upstream_request.extra_headers, body)
        log_request_summary(
            logger,
            provider_label="volcengine-ark-responses",
            model=upstream_request.model,
            messages=len(upstream_request.input),
            tools=len(upstream_request.tool_params()),
            extra=body,
            options=self.options,
        )
        connection, client_config = self._connection(request_model)
        resolution = PrefixCacheResolution(body=body)
        async with self.client.session(connection) as session:
            if self._prefix_cache_manager is not None:
                resolution = await self._prefix_cache_manager.resolve(
                    session,
                    body=body,
                    headers=request_headers,
                    query=upstream_request.extra_query,
                    client_config=client_config,
                )
                body = resolution.body
            resource_request = json_resource_request(body, headers=request_headers, query=upstream_request.extra_query)
            try:
                if stream:
                    payload = await collect_ark_response_stream(
                        session.responses.stream(resource_request, retry=session.retry),
                        model=upstream_request.model,
                    )
                else:
                    payload = await session.responses.create(resource_request, retry=session.retry)
            except ClientHttpError as exc:
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

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query, encoding_format = build_embedding_request(request_model, options=self.options)
        request_headers = build_ark_request_headers(extra_headers, body)
        connection, _ = self._connection(request_model)
        async with self.client.session(connection) as session:
            payload = await session.embeddings.create(
                json_resource_request(body, headers=request_headers, query=extra_query),
                retry=session.retry,
            )
        return build_ark_embedding_response(
            payload, options=self.options, encoding_format=encoding_format
        ).to_host_dict()

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query = build_ark_audio_transcription_request(
            request_model,
            options=self.options,
        )
        connection, _ = self._connection(request_model)
        async with self.client.session(connection) as session:
            payload = await session.audio_transcriptions.create(
                json_resource_request(body, headers=extra_headers, query=extra_query),
                retry=session.retry,
            )
        return parse_ark_audio_transcription_response(payload, options=self.options).to_host_dict()
