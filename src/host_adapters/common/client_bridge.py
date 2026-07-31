from collections.abc import Mapping

import httpx

from ...clients.common import HttpConnection, RetryPolicy, encode_query_value
from ...clients.families import JsonResourceRequest, MultipartResourceRequest
from ...core.json_types import JsonValue
from .httpx import HttpxClientConfig


def build_http_connection(config: HttpxClientConfig) -> HttpConnection:
    """把 Host 已解析的客户端配置冻结为 Client Connection。"""

    request_timeout: float | None
    connect_timeout: float | None
    if isinstance(config.timeout, httpx.Timeout):
        request_timeout = config.timeout.read
        connect_timeout = config.timeout.connect
    elif isinstance(config.timeout, (int, float)):
        request_timeout = float(config.timeout)
        connect_timeout = float(config.timeout)
    else:
        request_timeout = None
        connect_timeout = None
    return HttpConnection(
        base_url=config.base_url,
        default_headers=tuple(config.default_headers.items()),
        default_query=tuple(
            (key, encode_query_value(value)) for key, value in config.default_query.items() if value is not None
        ),
        request_timeout=request_timeout,
        connect_timeout=connect_timeout,
    )


def build_retry_policy(config: HttpxClientConfig, *, uncertain_on_timeout: bool = False) -> RetryPolicy:
    return RetryPolicy(
        max_retries=config.max_retries,
        retry_interval=config.retry_interval,
        uncertain_on_timeout=uncertain_on_timeout,
    )


def json_resource_request(
    body: Mapping[str, JsonValue],
    *,
    headers: Mapping[str, str],
    query: Mapping[str, object],
) -> JsonResourceRequest:
    # core 的 JsonValue 与 clients 的 ClientJsonValue 是两套按层封存、结构等价的递归别名，
    # pyright 直接判两者兼容，所以这里既不需要 cast 也不需要再做一次运行时窄化。
    return JsonResourceRequest(
        body=body,
        headers=headers,
        query=query,
    )


def multipart_resource_request(
    *,
    form_data: Mapping[str, str],
    files: Mapping[str, tuple[str, bytes]],
    headers: Mapping[str, str],
    query: Mapping[str, object],
) -> MultipartResourceRequest:
    return MultipartResourceRequest(
        form_data=form_data,
        files=files,
        headers=headers,
        query=query,
    )
