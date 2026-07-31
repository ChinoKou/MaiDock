import logging


from ...clients.siliconflow import SiliconFlowClient, SiliconFlowConnection
from ...core.common import ProviderRuntimeOptions, log_request_summary, log_response_summary
from ...core.json_types import json_list_or_none
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
from .audio_transcriptions import build_audio_transcription_request, parse_audio_transcription_response
from .chat import (
    SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT,
    build_chat_body,
    build_client_config,
    convert_response,
    resolve_path,
)
from .embeddings import (
    SILICONFLOW_EMBEDDINGS_ENDPOINT,
    build_embedding_request,
    build_siliconflow_embedding_response,
)
from .streaming import collect_stream_response
from ..common.options import build_host_common_options, build_siliconflow_host_options
from ..common.rpc import HostRpcRequest, HostRpcResponse

logger = logging.getLogger("maibot_plugin.maidock.siliconflow")

SILICONFLOW_AUDIO_TRANSCRIPTIONS_ENDPOINT = "audio/transcriptions"


class SiliconFlowHostAdapter:
    """MaiBot Host 合约到 SiliconFlow 原生资源的适配器。"""

    def __init__(self, *, options: ProviderRuntimeOptions, client: SiliconFlowClient) -> None:
        self.options = build_host_common_options(options)
        self.vendor_options = build_siliconflow_host_options(options)
        self.client = client

    def _connection(
        self,
        request: ResponseRequestSnapshot | EmbeddingRequestSnapshot | AudioTranscriptionRequestSnapshot,
    ) -> SiliconFlowConnection:
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
        return SiliconFlowConnection(
            http=build_http_connection(config),
            retry=build_retry_policy(config),
            chat_completions_path=resolve_path(config, SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT),
            embeddings_path=resolve_path(config, SILICONFLOW_EMBEDDINGS_ENDPOINT),
            audio_transcriptions_path=resolve_path(config, SILICONFLOW_AUDIO_TRANSCRIPTIONS_ENDPOINT),
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
        log_request_summary(
            logger,
            provider_label="siliconflow",
            model=str(body["model"]),
            messages=len(json_list_or_none(body.get("messages")) or []),
            tools=len(json_list_or_none(body.get("tools")) or []),
            extra=body,
            options=self.options,
        )
        connection = self._connection(request_model)
        resource_request = json_resource_request(body, headers=extra_headers, query=extra_query)
        async with self.client.session(connection) as session:
            if stream:
                result = await collect_stream_response(
                    session.chat_completions.stream(resource_request, retry=session.retry),
                    options=self.options,
                )
            else:
                payload = await session.chat_completions.create(resource_request, retry=session.retry)
                result = convert_response(payload, options=self.options)
        log_response_summary(
            logger,
            provider_label="siliconflow",
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
        return build_siliconflow_embedding_response(
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
