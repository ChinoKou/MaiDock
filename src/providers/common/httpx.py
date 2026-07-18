from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit
import asyncio
import httpx
import json
import logging

from ...core.common import (
    normalize_base_url,
    read_api_key,
    read_timeout,
    require_string_mapping,
    resolve_max_retries,
    resolve_retry_interval,
    with_default_user_agent,
)
from ...core.diagnostics import build_parse_error_message, sanitize_for_log
from ...core.json_types import json_mapping_or_none, mapping_to_json_object
from ...schemas.host_snapshots import ApiProviderSnapshot

_logger = logging.getLogger("maibot_plugin.maidock.httpx")

type HttpxTimeout = float | httpx.Timeout | None


@dataclass(slots=True)
class HttpxClientConfig:
    """从 MaiBot API Provider 快照派生的 httpx 客户端配置。"""

    base_url: str
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict = field(default_factory=dict)
    timeout: HttpxTimeout = None
    max_retries: int = 0
    retry_interval: float = 0.0


@dataclass(frozen=True, slots=True)
class SseJsonEvent:
    """从单个 SSE 事件解析出的 JSON 负载。"""

    event: str | None
    data: dict
    status: int | None = None


class HttpxProviderError(RuntimeError):
    """原生 httpx Provider 抛出的 Provider 错误。"""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HttpxProviderParseError(ValueError):
    """原生 httpx Provider 抛出的 Provider 响应解析错误。"""


def resolve_endpoint_path(base_url: str, *, api_prefix: str, endpoint_path: str) -> str:
    """返回相对路径，保留 base_url 中已有的 API 前缀。"""

    normalized_endpoint = endpoint_path.strip().strip("/")
    normalized_prefix = api_prefix.strip().strip("/")
    if not normalized_endpoint:
        raise ValueError("endpoint_path 不能为空")
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
    """构建通用的 Provider httpx 客户端配置。"""

    default_headers = require_string_mapping(api_provider.default_headers, field_name="api_provider.default_headers")
    default_query = api_provider.default_query.to_plain_dict()
    auth_type = (api_provider.auth_type or "bearer").strip().lower()
    api_key = read_api_key(api_provider, allow_empty=auth_type == "none")

    if auth_type in {"bearer", "header"}:
        default_headers[api_provider.auth_header_name] = _auth_header_value(api_provider.auth_header_prefix, api_key)
    elif auth_type == "query":
        default_query[api_provider.auth_query_name] = api_key
    elif auth_type != "none":
        raise ValueError(f"不支持的 auth_type: {api_provider.auth_type}")

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


def create_async_client(
    config: HttpxClientConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """从已验证的配置创建 AsyncClient。"""

    return httpx.AsyncClient(
        base_url=config.base_url,
        headers=config.default_headers,
        params=_query_params(config.default_query),
        timeout=config.timeout,
        transport=transport,
    )


def _query_params(query: Mapping[str, object] | None) -> dict[str, str] | None:
    if not query:
        return None
    result: dict[str, str] = {}
    for key, value in query.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[str(key)] = str(value)
        else:
            result[str(key)] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return result or None


def _headers(headers: Mapping[str, str] | None) -> dict[str, str] | None:
    return dict(headers) if headers else None


def _status_error_message(provider_label: str, response: httpx.Response) -> str:
    body: object
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return f"{provider_label} 上游接口返回状态码 {response.status_code}: {sanitize_for_log(body)}"


def _json_response_payload(provider_label: str, response: httpx.Response) -> dict:
    if response.status_code >= 400:
        raise HttpxProviderError(
            _status_error_message(provider_label, response),
            status_code=response.status_code,
        )
    try:
        raw_payload: object = response.json()
    except ValueError as exc:
        raise HttpxProviderParseError(build_parse_error_message(provider_label, "响应不是合法 JSON")) from exc
    payload = json_mapping_or_none(raw_payload)
    if payload is None:
        raise HttpxProviderParseError(build_parse_error_message(provider_label, "JSON 响应不是 object"))
    return mapping_to_json_object(payload)


def _header_retry_override(response: httpx.Response) -> bool | None:
    value = response.headers.get("x-should-retry")
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _should_retry_response(response: httpx.Response) -> bool:
    override = _header_retry_override(response)
    if override is not None:
        return override
    return response.status_code in {408, 409, 429} or response.status_code >= 500


async def post_json(
    client: httpx.AsyncClient,
    path: str,
    *,
    json_body: dict,
    headers: Mapping[str, str] | None = None,
    query: Mapping[str, object] | None = None,
    provider_label: str,
    max_retries: int = 0,
    retry_interval: float = 0.0,
) -> dict:
    """POST JSON 并返回已验证的 JSON object 响应。"""

    retries = max(max_retries, 0)
    for attempt in range(retries + 1):
        try:
            response = await client.post(
                path,
                json=json_body,
                headers=_headers(headers),
                params=_query_params(query),
            )
        except httpx.TimeoutException as exc:
            if attempt < retries:
                _logger.warning(
                    "[%s] POST 超时，将在 %.1f 秒后重试 (%d/%d): %s",
                    provider_label,
                    retry_interval,
                    attempt + 1,
                    retries,
                    exc,
                )
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)
                continue
            raise HttpxProviderError(f"{provider_label} 上游接口请求超时（已重试 {retries} 次）: {exc}") from exc
        except httpx.HTTPError as exc:
            if attempt < retries:
                _logger.warning(
                    "[%s] POST 连接失败，将在 %.1f 秒后重试 (%d/%d): %s",
                    provider_label,
                    retry_interval,
                    attempt + 1,
                    retries,
                    exc,
                )
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)
                continue
            raise HttpxProviderError(f"{provider_label} 上游接口连接失败（已重试 {retries} 次）: {exc}") from exc
        if response.status_code >= 400 and attempt < retries and _should_retry_response(response):
            _logger.warning(
                "[%s] POST 收到状态码 %d，将在 %.1f 秒后重试 (%d/%d)",
                provider_label,
                response.status_code,
                retry_interval,
                attempt + 1,
                retries,
            )
            await response.aclose()
            if retry_interval > 0:
                await asyncio.sleep(retry_interval)
            continue
        return _json_response_payload(provider_label, response)
    raise HttpxProviderError(f"{provider_label} 上游接口调用失败（已重试 {retries} 次）")


async def post_multipart(
    client: httpx.AsyncClient,
    path: str,
    *,
    form_data: Mapping[str, str],
    files: Mapping[str, tuple[str, bytes]],
    headers: Mapping[str, str] | None = None,
    query: Mapping[str, object] | None = None,
    provider_label: str,
    max_retries: int = 0,
    retry_interval: float = 0.0,
) -> httpx.Response:
    """POST multipart 表单数据并在状态码校验后返回原始响应。"""

    request_headers = dict(headers) if headers else {}
    for key in list(request_headers.keys()):
        if key.lower() == "content-type":
            request_headers.pop(key)

    retries = max(max_retries, 0)
    for attempt in range(retries + 1):
        try:
            response = await client.post(
                path,
                data=dict(form_data),
                files=dict(files),
                headers=request_headers or None,
                params=_query_params(query),
            )
        except httpx.TimeoutException as exc:
            if attempt < retries:
                _logger.warning(
                    "[%s] POST multipart 超时，将在 %.1f 秒后重试 (%d/%d): %s",
                    provider_label,
                    retry_interval,
                    attempt + 1,
                    retries,
                    exc,
                )
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)
                continue
            raise HttpxProviderError(f"{provider_label} 上游接口请求超时（已重试 {retries} 次）: {exc}") from exc
        except httpx.HTTPError as exc:
            if attempt < retries:
                _logger.warning(
                    "[%s] POST multipart 连接失败，将在 %.1f 秒后重试 (%d/%d): %s",
                    provider_label,
                    retry_interval,
                    attempt + 1,
                    retries,
                    exc,
                )
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)
                continue
            raise HttpxProviderError(f"{provider_label} 上游接口连接失败（已重试 {retries} 次）: {exc}") from exc
        if response.status_code >= 400 and attempt < retries and _should_retry_response(response):
            _logger.warning(
                "[%s] POST multipart 收到状态码 %d，将在 %.1f 秒后重试 (%d/%d)",
                provider_label,
                response.status_code,
                retry_interval,
                attempt + 1,
                retries,
            )
            await response.aclose()
            if retry_interval > 0:
                await asyncio.sleep(retry_interval)
            continue
        if response.status_code >= 400:
            raise HttpxProviderError(
                _status_error_message(provider_label, response),
                status_code=response.status_code,
            )
        return response
    raise HttpxProviderError(f"{provider_label} 上游接口调用失败（已重试 {retries} 次）")


def _parse_sse_data(provider_label: str, data: str) -> dict | None:
    normalized = data.strip()
    if not normalized or normalized == "[DONE]":
        return None
    try:
        raw_payload: object = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise HttpxProviderParseError(build_parse_error_message(provider_label, "SSE data 不是合法 JSON")) from exc
    payload = json_mapping_or_none(raw_payload)
    if payload is None:
        raise HttpxProviderParseError(build_parse_error_message(provider_label, "SSE JSON data 不是 object"))
    return mapping_to_json_object(payload)


def _extract_sse_value(line: str, prefix: str) -> str:
    value = line[len(prefix) :]
    if value.startswith(" "):
        value = value[1:]
    return value


def _parse_sse_status(line: str) -> int | None:
    value = _extract_sse_value(line, "status:").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


async def stream_sse_json(
    client: httpx.AsyncClient,
    path: str,
    *,
    json_body: dict,
    headers: Mapping[str, str] | None = None,
    query: Mapping[str, object] | None = None,
    provider_label: str,
    max_retries: int = 0,
    retry_interval: float = 0.0,
) -> AsyncIterator[SseJsonEvent]:
    """POST JSON 并解析 JSON SSE 事件。"""

    retries = max(max_retries, 0)
    for attempt in range(retries + 1):
        emitted_event = False
        try:
            async with client.stream(
                "POST",
                path,
                json=json_body,
                headers=_headers(headers),
                params=_query_params(query),
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    if attempt < retries and _should_retry_response(response):
                        _logger.warning(
                            "[%s] SSE 流收到状态码 %d，将在 %.1f 秒后重试 (%d/%d)",
                            provider_label,
                            response.status_code,
                            retry_interval,
                            attempt + 1,
                            retries,
                        )
                        if retry_interval > 0:
                            await asyncio.sleep(retry_interval)
                        continue
                    raise HttpxProviderError(
                        _status_error_message(provider_label, response),
                        status_code=response.status_code,
                    )
                event_name: str | None = None
                status: int | None = None
                data_lines: list[str] = []
                async for raw_line in response.aiter_lines():
                    line = raw_line.rstrip("\r")
                    if not line:
                        payload = _parse_sse_data(provider_label, "\n".join(data_lines))
                        if payload is None:
                            if data_lines and "\n".join(data_lines).strip() == "[DONE]":
                                break
                        else:
                            emitted_event = True
                            yield SseJsonEvent(event=event_name, data=payload, status=status)
                        event_name = None
                        status = None
                        data_lines.clear()
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = _extract_sse_value(line, "event:")
                    elif line.startswith("status:"):
                        status = _parse_sse_status(line)
                    elif line.startswith("data:"):
                        data_lines.append(_extract_sse_value(line, "data:"))
                if data_lines:
                    payload = _parse_sse_data(provider_label, "\n".join(data_lines))
                    if payload is not None:
                        emitted_event = True
                        yield SseJsonEvent(event=event_name, data=payload, status=status)
                return
        except httpx.TimeoutException as exc:
            if not emitted_event and attempt < retries:
                _logger.warning(
                    "[%s] SSE 流超时，将在 %.1f 秒后重试 (%d/%d): %s",
                    provider_label,
                    retry_interval,
                    attempt + 1,
                    retries,
                    exc,
                )
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)
                continue
            raise HttpxProviderError(f"{provider_label} 上游接口请求超时（已重试 {retries} 次）: {exc}") from exc
        except httpx.HTTPError as exc:
            if not emitted_event and attempt < retries:
                _logger.warning(
                    "[%s] SSE 流连接失败，将在 %.1f 秒后重试 (%d/%d): %s",
                    provider_label,
                    retry_interval,
                    attempt + 1,
                    retries,
                    exc,
                )
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)
                continue
            raise HttpxProviderError(f"{provider_label} 上游接口连接失败（已重试 {retries} 次）: {exc}") from exc
