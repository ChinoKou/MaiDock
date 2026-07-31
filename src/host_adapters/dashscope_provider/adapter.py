import logging

from ...clients.dashscope import (
    DashScopeClient,
    DashScopeClientError,
    DashScopeConnection,
    DashScopePaths,
)
from ...core.common import ProviderRuntimeOptions, log_request_summary, log_response_summary, read_model_identifier
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..common.client_bridge import build_http_connection, build_retry_policy, json_resource_request
from .audio_transcriptions import build_audio_transcription_request, parse_audio_transcription_response
from .chat import (
    DASHSCOPE_GENERATION_ENDPOINT,
    DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT,
    DashScopeEndpointKind,
    build_client_config,
    build_generation_body,
    convert_response,
    count_tools,
    is_dashscope_multimodal_model,
    is_dashscope_text_only_model,
    normalize_dashscope_model,
    request_has_image,
    resolve_path,
)
from .embeddings import build_dashscope_embedding_response, build_embedding_request
from .errors import DashScopeApiError
from .streaming import collect_stream_response
from ..common.options import build_dashscope_host_options, build_host_common_options
from ..common.rpc import HostRpcRequest, HostRpcResponse
from ...core.json_types import JsonValue

logger = logging.getLogger("maibot_plugin.maidock.dashscope")

_IMAGE_GENERATION_ENDPOINT = "services/aigc/image-generation/generation"
_TEXT2IMAGE_SYNTHESIS_ENDPOINT = "services/aigc/text2image/image-synthesis"
_IMAGE2IMAGE_SYNTHESIS_ENDPOINT = "services/aigc/image2image/image-synthesis"
_VIDEO_GENERATION_ENDPOINT = "services/aigc/video-generation/video-synthesis"


class DashScopeHostAdapter:
    """MaiBot Host 合约到 DashScope 原生资源的适配器。"""

    def __init__(self, *, options: ProviderRuntimeOptions, client: DashScopeClient) -> None:
        self.options = build_host_common_options(options)
        self.vendor_options = build_dashscope_host_options(options)
        self.client = client
        self._endpoint_cache: dict[str, DashScopeEndpointKind] = {}

    def _resolve_chat_endpoint_kind(
        self,
        request: ResponseRequestSnapshot,
        model: str,
        *,
        has_image: bool,
    ) -> DashScopeEndpointKind:
        if has_image and is_dashscope_text_only_model(model):
            from ...i18n import translate

            raise ValueError(translate("runtime.error.dashscope_visual_unsupported", model=model))
        if has_image:
            return "multimodal"
        normalized_model = normalize_dashscope_model(model)
        cached = self._endpoint_cache.get(normalized_model)
        if cached is not None:
            return cached
        if is_dashscope_text_only_model(model):
            return "text"
        if is_dashscope_multimodal_model(model) or request.model_info.visual:
            return "multimodal"
        return "text"

    def _connection(
        self,
        request: ResponseRequestSnapshot | EmbeddingRequestSnapshot | AudioTranscriptionRequestSnapshot,
        *,
        embeddings_endpoint: str,
    ) -> DashScopeConnection:
        connection_options = self.vendor_options.connection
        config = build_client_config(
            request.api_provider,
            user_agent=connection_options.user_agent,
            force_official_endpoint=self.vendor_options.force_official_endpoint,
            default_max_retries=connection_options.max_retries,
            force_max_retries=connection_options.force_max_retries,
            default_retry_interval=connection_options.retry_interval,
            force_retry_interval=connection_options.force_retry_interval,
        )
        return DashScopeConnection(
            http=build_http_connection(config),
            retry=build_retry_policy(config),
            safe_retry=build_retry_policy(config),
            paths=DashScopePaths(
                text_generation=resolve_path(config, DASHSCOPE_GENERATION_ENDPOINT),
                multimodal_generation=resolve_path(config, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
                embeddings=resolve_path(config, embeddings_endpoint),
                image_generation=resolve_path(config, _IMAGE_GENERATION_ENDPOINT),
                text2image_synthesis=resolve_path(config, _TEXT2IMAGE_SYNTHESIS_ENDPOINT),
                image2image_synthesis=resolve_path(config, _IMAGE2IMAGE_SYNTHESIS_ENDPOINT),
                video_generation=resolve_path(config, _VIDEO_GENERATION_ENDPOINT),
                tasks=resolve_path(config, "tasks"),
                uploads=resolve_path(config, "uploads"),
            ),
        )

    @staticmethod
    def _host_error(error: DashScopeClientError) -> DashScopeApiError:
        return DashScopeApiError(
            str(error),
            code=error.code,
            upstream_message=error.upstream_message,
            request_id=error.request_id,
            status_code=error.status_code,
        )

    async def _request_chat(
        self,
        session,
        endpoint_kind: DashScopeEndpointKind,
        body: dict[str, JsonValue],
        *,
        stream: bool,
        headers: dict[str, str],
        query: dict[str, JsonValue],
    ) -> ProviderResponse:
        resource = session.text_generation if endpoint_kind == "text" else session.multimodal_generation
        request = json_resource_request(body, headers=headers, query=query)
        try:
            if stream:
                stream_headers = {
                    **headers,
                    "Accept": "text/event-stream",
                    "X-Accel-Buffering": "no",
                    "X-DashScope-SSE": "enable",
                }
                events = resource.stream(
                    json_resource_request(body, headers=stream_headers, query=query),
                    retry=session.retry,
                )
                return await collect_stream_response(
                    events,
                    options=self.options,
                    is_multimodal=endpoint_kind == "multimodal",
                )
            payload = await resource.create(request, retry=session.retry)
        except DashScopeClientError as exc:
            raise self._host_error(exc) from exc
        return convert_response(payload, options=self.options, is_multimodal=endpoint_kind == "multimodal")

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = ResponseRequestSnapshot.model_validate(request)
        model = read_model_identifier(request_model.model_info)
        has_image = request_has_image(request_model.message_list)
        endpoint_kind = self._resolve_chat_endpoint_kind(request_model, model, has_image=has_image)
        stream = bool(request_model.model_info.force_stream_mode)
        body, extra_headers, extra_query = build_generation_body(
            request_model,
            options=self.options,
            stream=stream,
            logger=logger,
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
        connection = self._connection(request_model, embeddings_endpoint=DASHSCOPE_GENERATION_ENDPOINT)
        async with self.client.session(connection) as session:
            try:
                result = await self._request_chat(
                    session,
                    endpoint_kind,
                    body,
                    stream=stream,
                    headers=extra_headers,
                    query=extra_query,
                )
            except DashScopeApiError as exc:
                if not (self.vendor_options.auto_detect_endpoint and not has_image and exc.is_endpoint_mismatch):
                    raise
                alternate_kind: DashScopeEndpointKind = "multimodal" if endpoint_kind == "text" else "text"
                result = await self._request_chat(
                    session,
                    alternate_kind,
                    body,
                    stream=stream,
                    headers=extra_headers,
                    query=extra_query,
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

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        endpoint, body, extra_headers, extra_query = build_embedding_request(request_model, options=self.options)
        connection = self._connection(request_model, embeddings_endpoint=endpoint)
        async with self.client.session(connection) as session:
            try:
                payload = await session.embeddings.create(
                    json_resource_request(body, headers=extra_headers, query=extra_query),
                    retry=session.retry,
                )
            except DashScopeClientError as exc:
                raise self._host_error(exc) from exc
        return build_dashscope_embedding_response(payload, options=self.options).to_host_dict()

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query = build_audio_transcription_request(request_model, options=self.options)
        connection = self._connection(request_model, embeddings_endpoint=DASHSCOPE_GENERATION_ENDPOINT)
        async with self.client.session(connection) as session:
            try:
                payload = await session.audio_transcriptions.create(
                    json_resource_request(body, headers=extra_headers, query=extra_query),
                    retry=session.retry,
                )
            except DashScopeClientError as exc:
                raise self._host_error(exc) from exc
        content, raw_data = parse_audio_transcription_response(payload, options=self.options)
        return ProviderResponse(content=content, raw_data=raw_data).to_host_dict()
