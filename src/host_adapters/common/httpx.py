from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from ...core.json_types import JsonValue
from ...core.common import (
    normalize_auth_type,
    normalize_base_url,
    read_api_key,
    read_timeout,
    require_string_mapping,
    resolve_max_retries,
    resolve_retry_interval,
    with_default_user_agent,
)
from ...i18n import runtime_item, translate
from ...schemas.host_snapshots import ApiProviderSnapshot

type HttpxTimeout = float | httpx.Timeout | None


@dataclass(slots=True)
class HttpxClientConfig:
    """从 Host API Provider 快照派生的连接构造输入。"""

    base_url: str
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    timeout: HttpxTimeout = None
    max_retries: int = 0
    retry_interval: float = 0.0


class HttpxProviderError(RuntimeError):
    """Host 结果映射阶段使用的供应商错误。"""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HttpxProviderParseError(ValueError):
    """Host 结果映射阶段使用的响应解析错误。"""


def resolve_endpoint_path(base_url: str, *, api_prefix: str, endpoint_path: str) -> str:
    """返回相对路径，保留 base_url 中已有的 API 前缀。"""

    normalized_endpoint = endpoint_path.strip().strip("/")
    normalized_prefix = api_prefix.strip().strip("/")
    if not normalized_endpoint:
        raise ValueError(
            translate("runtime.error.required", subject="endpoint_path", field=runtime_item("non_empty_path"))
        )
    if not normalized_prefix:
        return normalized_endpoint

    base_path = urlsplit(base_url).path.strip("/")
    if base_path == normalized_prefix or base_path.endswith(f"/{normalized_prefix}"):
        return normalized_endpoint
    return f"{normalized_prefix}/{normalized_endpoint}"


def _auth_header_value(prefix: str, api_key: str) -> str:
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        return api_key
    return f"{normalized_prefix} {api_key}"


def _resolve_httpx_base_url(
    api_provider: ApiProviderSnapshot,
    *,
    default_base_url: str,
    force_default_base_url: bool,
) -> str:
    if force_default_base_url:
        return normalize_base_url(default_base_url)
    return normalize_base_url(api_provider.base_url)


def build_httpx_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    default_base_url: str,
    user_agent: str | None,
    force_default_base_url: bool = False,
    default_timeout: HttpxTimeout = None,
    default_max_retries: int = 0,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    """按既有优先级构建 Host 侧连接输入。"""

    default_headers = require_string_mapping(api_provider.default_headers, field_name="api_provider.default_headers")
    default_query = api_provider.default_query.to_plain_dict()
    auth_type = normalize_auth_type(api_provider.auth_type)
    api_key = read_api_key(api_provider, allow_empty=auth_type == "none")

    if auth_type in {"bearer", "header"}:
        default_headers[api_provider.auth_header_name] = _auth_header_value(api_provider.auth_header_prefix, api_key)
    elif auth_type == "query":
        default_query[api_provider.auth_query_name] = api_key
    normalized_base_url = _resolve_httpx_base_url(
        api_provider,
        default_base_url=default_base_url,
        force_default_base_url=force_default_base_url,
    )

    headers = with_default_user_agent(default_headers, user_agent)
    headers.setdefault("Accept", "application/json")
    headers.setdefault("Content-Type", "application/json")
    timeout = read_timeout(api_provider)

    return HttpxClientConfig(
        base_url=normalized_base_url,
        default_headers=headers,
        default_query=default_query,
        timeout=timeout if timeout is not None else default_timeout,
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
