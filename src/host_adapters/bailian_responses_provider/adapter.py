import logging
from urllib.parse import urlsplit

from ...clients.dashscope import DashScopeClient, DashScopeResponsesConnection
from ...core.common import (
    ProviderRuntimeOptions,
    build_openai_compatible_client_config,
    log_request_summary,
    log_response_summary,
    normalize_base_url,
    read_timeout,
    resolve_max_retries,
    resolve_retry_interval,
)
from ...i18n import translate
from ...schemas import ApiProviderSnapshot, ResponseRequestSnapshot
from ..common.client_bridge import build_http_connection, build_retry_policy, json_resource_request
from ..common.options import build_bailian_host_options, build_host_common_options
from ..common.rpc import HostRpcRequest, HostRpcResponse
from ..responses_family.transport import HttpxClientConfig, resolve_endpoint_path
from .responses import create_responses_mapper
from .streaming import collect_bailian_response_stream

logger = logging.getLogger("maibot_plugin.maidock.bailian_responses")


def bailian_responses_path(base_url: str) -> str:
    """校验百炼 Responses base URL（必须以 /v1 结尾），返回追加的 /responses 路径。

    错误地填写完整 endpoint（以 /responses 结尾）、DashScope 原生地址（/api/v1）
    或其他形式时明确报错。
    """

    normalized = normalize_base_url(base_url)
    path = urlsplit(normalized).path.rstrip("/")
    if path.endswith("/responses"):
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="Bailian Responses base_url",
                allowed="以 /v1 结尾的 base URL（例如 https://dashscope.aliyuncs.com/compatible-mode/v1），不要填写完整 /responses 端点",
            )
        )
    if path.endswith("/api/v1"):
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="Bailian Responses base_url",
                allowed="以 /v1 结尾的 OpenAI 兼容 base URL，不要填写 DashScope 原生 /api/v1 地址",
            )
        )
    if not path.endswith("/v1"):
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="Bailian Responses base_url",
                allowed="以 /v1 结尾的 OpenAI 兼容 base URL（例如 https://dashscope.aliyuncs.com/compatible-mode/v1）",
            )
        )
    return resolve_endpoint_path(normalized, api_prefix="v1", endpoint_path="responses")


def build_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str,
    default_max_retries: int = 3,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    """构造百炼 Responses 连接配置；base_url 必须合法且以 /v1 结尾。"""

    base_url = normalize_base_url(api_provider.base_url)
    bailian_responses_path(base_url)
    client_config = build_openai_compatible_client_config(api_provider, user_agent=user_agent)
    headers = dict(client_config.default_headers)
    if client_config.api_key:
        headers.setdefault("Authorization", f"Bearer {client_config.api_key}")
    headers.setdefault("Accept", "application/json")
    return HttpxClientConfig(
        base_url=base_url,
        default_headers=headers,
        default_query=dict(client_config.default_query),
        timeout=read_timeout(api_provider),
        max_retries=resolve_max_retries(
            api_provider,
            config_value=default_max_retries,
            force=force_max_retries,
            default=3,
        ),
        retry_interval=resolve_retry_interval(
            api_provider,
            config_value=default_retry_interval,
            force=force_retry_interval,
            default=5.0,
        ),
    )


class BailianResponsesHostAdapter:
    """MaiBot Host 合约到百炼 Responses 原生资源的适配器。"""

    def __init__(self, *, options: ProviderRuntimeOptions, client: DashScopeClient) -> None:
        self.options = build_host_common_options(options)
        self.vendor_options = build_bailian_host_options(options)
        self.client = client
        self._responses_mapper = create_responses_mapper(options=self.options, logger=logger)

    def _connection(self, request: ResponseRequestSnapshot) -> DashScopeResponsesConnection:
        connection_options = self.vendor_options.connection
        config = build_client_config(
            request.api_provider,
            user_agent=connection_options.user_agent,
            default_max_retries=connection_options.max_retries,
            force_max_retries=connection_options.force_max_retries,
            default_retry_interval=connection_options.retry_interval,
            force_retry_interval=connection_options.force_retry_interval,
        )
        return DashScopeResponsesConnection(
            http=build_http_connection(config),
            retry=build_retry_policy(config),
            responses_path=bailian_responses_path(config.base_url),
        )

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
        del request
        raise ValueError(translate("runtime.error.operation_unsupported", operation="embedding"))

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        del request
        raise ValueError(translate("runtime.error.operation_unsupported", operation="audio_transcription"))

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
        request_model = ResponseRequestSnapshot.model_validate(request)
        upstream_request = self._responses_mapper.build_request(request_model)
        stream = bool(request_model.model_info.force_stream_mode)
        body = self._responses_mapper.build_http_body(upstream_request, stream=stream)
        log_request_summary(
            logger,
            provider_label="bailian-responses",
            model=upstream_request.model,
            messages=len(upstream_request.input),
            tools=len(upstream_request.tool_params()),
            extra=body,
            options=self.options,
        )
        connection = self._connection(request_model)
        resource_request = json_resource_request(
            body,
            headers=upstream_request.extra_headers,
            query=upstream_request.extra_query,
        )
        async with self.client.responses_session(connection) as session:
            if stream:
                payload = await collect_bailian_response_stream(
                    session.responses.stream(resource_request, retry=session.retry),
                    model=upstream_request.model,
                )
            else:
                payload = await session.responses.create(resource_request, retry=session.retry)
        result = self._responses_mapper.convert_response(payload)
        log_response_summary(
            logger,
            provider_label="bailian-responses",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()
