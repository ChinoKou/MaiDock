import logging

import httpx
from maibot_sdk import LLMProviderBase

from ...core.common import (
    ProviderRuntimeOptions,
    log_request_summary,
    log_response_summary,
)
from ...schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..openai_auxiliary_family.transport import (
    create_async_client as create_auxiliary_client,
)
from ..openai_auxiliary_family.transport import (
    post_json as post_auxiliary_json,
)
from ..openai_auxiliary_family.transport import (
    post_multipart,
)
from ..openai_auxiliary_family.transport import (
    resolve_endpoint_path as resolve_auxiliary_endpoint_path,
)
from ..responses_family.transport import (
    create_async_client as create_responses_client,
)
from ..responses_family.transport import (
    post_json as post_responses_json,
)
from ..responses_family.transport import (
    resolve_endpoint_path as resolve_responses_endpoint_path,
)
from .audio_transcriptions import (
    build_audio_transcription_request,
    parse_audio_transcription_response,
)
from .embeddings import (
    build_embedding_request,
    build_openai_embedding_response,
    extract_openai_embedding,
)
from .responses import (
    OPENAI_API_PREFIX,
    OPENAI_AUDIO_TRANSCRIPTIONS_ENDPOINT,
    OPENAI_EMBEDDINGS_ENDPOINT,
    OPENAI_PROVIDER_LABEL,
    OPENAI_RESPONSES_ENDPOINT,
    build_client_config,
    create_responses_mapper,
)
from .streaming import collect_openai_response_stream

logger = logging.getLogger("maibot_plugin.maidock.openai_responses")


class OpenAIResponsesProvider(LLMProviderBase):
    """基于 OpenAI Responses API 的原生 HTTP Provider。"""

    _extract_embedding = staticmethod(extract_openai_embedding)

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

        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.openai_user_agent,
            default_max_retries=self.options.openai_max_retries,
            force_max_retries=self.options.openai_force_max_retries,
            default_retry_interval=self.options.openai_retry_interval,
            force_retry_interval=self.options.openai_force_retry_interval,
        )
        path = resolve_responses_endpoint_path(
            config.base_url,
            api_prefix=OPENAI_API_PREFIX,
            endpoint_path=OPENAI_RESPONSES_ENDPOINT,
        )
        async with create_responses_client(config, transport=self._transport) as client:
            if stream:
                payload = await collect_openai_response_stream(
                    client,
                    path,
                    body,
                    headers=upstream_request.extra_headers,
                    query=upstream_request.extra_query,
                    model=upstream_request.model,
                    max_retries=config.max_retries,
                    retry_interval=config.retry_interval,
                )
            else:
                payload = await post_responses_json(
                    client,
                    path,
                    json_body=body,
                    headers=upstream_request.extra_headers,
                    query=upstream_request.extra_query,
                    provider_label=OPENAI_PROVIDER_LABEL,
                    max_retries=config.max_retries,
                    retry_interval=config.retry_interval,
                )

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

    async def get_embedding(self, request: dict) -> dict:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        body, extra_headers, extra_query, encoding_format = build_embedding_request(request_model, options=self.options)
        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.openai_user_agent,
            default_max_retries=self.options.openai_max_retries,
            force_max_retries=self.options.openai_force_max_retries,
            default_retry_interval=self.options.openai_retry_interval,
            force_retry_interval=self.options.openai_force_retry_interval,
        )
        path = resolve_auxiliary_endpoint_path(
            config.base_url,
            api_prefix=OPENAI_API_PREFIX,
            endpoint_path=OPENAI_EMBEDDINGS_ENDPOINT,
        )
        async with create_auxiliary_client(config, transport=self._transport) as client:
            payload = await post_auxiliary_json(
                client,
                path,
                json_body=body,
                headers=extra_headers,
                query=extra_query,
                provider_label="OpenAI Embeddings",
                max_retries=config.max_retries,
                retry_interval=config.retry_interval,
            )

        return build_openai_embedding_response(
            payload, options=self.options, encoding_format=encoding_format
        ).to_host_dict()

    async def get_audio_transcriptions(self, request: dict) -> dict:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        form_data, files, extra_headers, extra_query = build_audio_transcription_request(
            request_model,
            options=self.options,
        )
        config = build_client_config(
            request_model.api_provider,
            user_agent=self.options.openai_user_agent,
            default_max_retries=self.options.openai_max_retries,
            force_max_retries=self.options.openai_force_max_retries,
            default_retry_interval=self.options.openai_retry_interval,
            force_retry_interval=self.options.openai_force_retry_interval,
        )
        path = resolve_auxiliary_endpoint_path(
            config.base_url,
            api_prefix=OPENAI_API_PREFIX,
            endpoint_path=OPENAI_AUDIO_TRANSCRIPTIONS_ENDPOINT,
        )
        async with create_auxiliary_client(config, transport=self._transport) as client:
            response = await post_multipart(
                client,
                path,
                form_data=form_data,
                files=files,
                headers=extra_headers,
                query=extra_query,
                provider_label="OpenAI Audio Transcriptions",
                max_retries=config.max_retries,
                retry_interval=config.retry_interval,
            )

        content, raw_data = parse_audio_transcription_response(response, options=self.options)
        return ProviderResponse(content=content, raw_data=raw_data).to_host_dict()
