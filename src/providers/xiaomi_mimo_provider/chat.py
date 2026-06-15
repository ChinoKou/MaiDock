import logging

from ...core.common import (
    ProviderRuntimeOptions,
    read_api_key,
    read_timeout,
    resolve_max_retries,
    resolve_retry_interval,
)
from ...schemas import (
    ApiProviderSnapshot,
    ProviderResponse,
    ResponseRequestSnapshot,
)
from ..chat_completions_family.chat import ChatCompletionsMapper
from ..common.httpx import (
    HttpxClientConfig,
    normalize_base_url,
    resolve_endpoint_path,
    with_default_user_agent,
)

MIMO_PROVIDER_LABEL = "Xiaomi Mimo"
MIMO_CHAT_COMPLETIONS_ENDPOINT = "chat/completions"


def build_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str,
    default_max_retries: int = 3,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    api_key = read_api_key(api_provider)
    default_headers = {}
    if api_provider.default_headers:
        for key, value in api_provider.default_headers.fields.items():
            if isinstance(value, str):
                default_headers[str(key)] = value
    default_headers["api-key"] = api_key
    default_headers = with_default_user_agent(default_headers, user_agent)
    default_headers.setdefault("Accept", "application/json")
    default_headers.setdefault("Content-Type", "application/json")

    base_url = normalize_base_url(api_provider.base_url)

    default_query = api_provider.default_query.to_plain_dict()

    return HttpxClientConfig(
        base_url=base_url,
        default_headers=default_headers,
        default_query=default_query,
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


def resolve_path(config: HttpxClientConfig, endpoint: str) -> str:
    return resolve_endpoint_path(config.base_url, api_prefix="", endpoint_path=endpoint)


def _create_mapper(*, options: ProviderRuntimeOptions, logger: logging.Logger) -> ChatCompletionsMapper:
    return ChatCompletionsMapper(
        options=options,
        logger=logger,
        provider_label=MIMO_PROVIDER_LABEL,
        raw_provider="xiaomi_mimo",
        policy_provider="xiaomi_mimo",
        tool_namespace="xiaomi_mimo",
        history_tool_prefix="mimo_history_tool",
        extract_tool_prefix="mimo_tool",
    )


def build_chat_body(
    request: ResponseRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
    logger: logging.Logger,
    stream: bool,
) -> tuple[dict, dict[str, str], dict]:
    mapper = _create_mapper(options=options, logger=logger)
    body, headers, query = mapper.build_request_body(request, stream=stream)
    if options.mimo_force_disable_thinking:
        body["thinking"] = {"type": "disabled"}
    return body, headers, query


def convert_response(payload: dict, *, options: ProviderRuntimeOptions) -> ProviderResponse:
    mapper = _create_mapper(options=options, logger=logging.getLogger(__name__))
    return mapper.convert_response(payload)
