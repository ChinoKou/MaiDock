import logging


from ...clients.openai import OpenAIClient, OpenAIConnection
from ...core.common import ProviderRuntimeOptions, log_request_summary, log_response_summary
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..common.client_bridge import (
    build_http_connection,
    build_retry_policy,
    json_resource_request,
    multipart_resource_request,
)
from ..common.httpx import resolve_endpoint_path
from ..common.options import build_host_common_options, build_openai_host_options
from ..common.rpc import HostRpcRequest, HostRpcResponse
from .audio_transcriptions import build_audio_transcription_request, parse_audio_transcription_response
from .embeddings import build_embedding_request, build_openai_embedding_response, extract_openai_embedding
from .responses import (
    OPENAI_API_PREFIX,
    OPENAI_AUDIO_TRANSCRIPTIONS_ENDPOINT,
    OPENAI_EMBEDDINGS_ENDPOINT,
    OPENAI_RESPONSES_ENDPOINT,
    build_client_config,
    create_responses_mapper,
)
from .streaming import collect_openai_response_stream

logger = logging.getLogger("maibot_plugin.maidock.openai_responses")


class OpenAIHostAdapter:
    """MaiBot Host 合约到 OpenAI 原生资源的适配器。"""

    _extract_embedding = staticmethod(extract_openai_embedding)

    def __init__(self, *, options: ProviderRuntimeOptions, client: OpenAIClient) -> None:
        self.options = build_host_common_options(options)
        self.vendor_options = build_openai_host_options(options)
        self.client = client
        self._responses_mapper = create_responses_mapper(options=self.options, logger=logger)

    def _connection(
        self, request: ResponseRequestSnapshot | EmbeddingRequestSnapshot | AudioTranscriptionRequestSnapshot
    ) -> OpenAIConnection:
        connection_options = self.vendor_options.connection
        config = build_client_config(
            request.api_provider,
            user_agent=connection_options.user_agent,
            default_max_retries=connection_options.max_retries,
            force_max_retries=connection_options.force_max_retries,
            default_retry_interval=connection_options.retry_interval,
            force_retry_interval=connection_options.force_retry_interval,
        )
        return OpenAIConnection(
            http=build_http_connection(config),
            retry=build_retry_policy(config),
            responses_path=resolve_endpoint_path(
                config.base_url,
                api_prefix=OPENAI_API_PREFIX,
                endpoint_path=OPENAI_RESPONSES_ENDPOINT,
            ),
            embeddings_path=resolve_endpoint_path(
                config.base_url,
                api_prefix=OPENAI_API_PREFIX,
                endpoint_path=OPENAI_EMBEDDINGS_ENDPOINT,
            ),
            audio_transcriptions_path=resolve_endpoint_path(
                config.base_url,
                api_prefix=OPENAI_API_PREFIX,
                endpoint_path=OPENAI_AUDIO_TRANSCRIPTIONS_ENDPOINT,
            ),
        )

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = ResponseRequestSnapshot.model_validate(request)
        upstream_request = self._responses_mapper.build_request(request_model)
        stream = bool(request_model.model_info.force_stream_mode)
        body = self._responses_mapper.build_http_body(upstream_request, stream=stream)
        log_request_summary(
            logger,
            provider_label="openai-responses",
            model=upstream_request.model,
            messages=len(upstream_request.input),
            tools=len(upstream_request.tools),
            extra=body,
            options=self.options,
        )
        connection = self._connection(request_model)
        resource_request = json_resource_request(
            body,
            headers=upstream_request.extra_headers,
            query=upstream_request.extra_query,
        )
        async with self.client.session(connection) as session:
            if stream:
                payload = await collect_openai_response_stream(
                    session.responses.stream(resource_request, retry=session.retry),
                    model=upstream_request.model,
                )
            else:
                payload = await session.responses.create(resource_request, retry=session.retry)
        result = self._responses_mapper.convert_response(payload)
        log_response_summary(
            logger,
            provider_label="openai-responses",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query, encoding_format = build_embedding_request(
            request_model,
            options=self.options,
        )
        connection = self._connection(request_model)
        async with self.client.session(connection) as session:
            payload = await session.embeddings.create(
                json_resource_request(body, headers=extra_headers, query=extra_query),
                retry=session.retry,
            )
        return build_openai_embedding_response(
            payload,
            options=self.options,
            encoding_format=encoding_format,
        ).to_host_dict()

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        form_data, files, extra_headers, extra_query = build_audio_transcription_request(
            request_model,
            options=self.options,
        )
        connection = self._connection(request_model)
        async with self.client.session(connection) as session:
            response = await session.audio_transcriptions.create(
                multipart_resource_request(
                    form_data=form_data,
                    files=files,
                    headers=extra_headers,
                    query=extra_query,
                ),
                retry=session.retry,
            )
        content, raw_data = parse_audio_transcription_response(response, options=self.options)
        return ProviderResponse(content=content, raw_data=raw_data).to_host_dict()
