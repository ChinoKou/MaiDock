from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

import asyncio
import json
import os

import httpx

from .types import JsonObject, JsonValue


class ClientHttpError(RuntimeError):
    """供应商 Client 的 HTTP 错误。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        uncertain: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.uncertain = uncertain


class ClientTimeoutError(ClientHttpError):
    """上游调用超时。"""


class ClientConnectionError(ClientHttpError):
    """上游连接失败。"""


class ClientProtocolError(ClientHttpError):
    """上游响应不符合协议。"""


class ClientClosedError(RuntimeError):
    """Client 已停止接收新 Session。"""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """由具体资源声明的重试策略。"""

    max_retries: int = 0
    retry_interval: float = 0.0
    retry_timeouts: bool = True
    retry_connections: bool = True
    retry_statuses: frozenset[int] = frozenset({408, 409, 429})
    retry_server_errors: bool = True
    uncertain_on_timeout: bool = False

    def should_retry_status(self, status_code: int) -> bool:
        return status_code in self.retry_statuses or self.retry_server_errors and status_code >= 500


NO_RETRY = RetryPolicy()


@dataclass(frozen=True, slots=True)
class HttpConnection:
    """单次供应商调用使用的不可变连接快照。"""

    base_url: str
    default_headers: tuple[tuple[str, str], ...] = ()
    default_query: tuple[tuple[str, str], ...] = ()
    request_timeout: float | None = None
    connect_timeout: float | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url 必须是有效的 HTTP 或 HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url 不允许包含用户凭据")

    def headers(self) -> dict[str, str]:
        return dict(self.default_headers)

    def query(self) -> dict[str, str]:
        return dict(self.default_query)


@dataclass(frozen=True, slots=True)
class SseJsonEvent:
    event: str | None
    data: JsonObject
    status: int | None = None


@dataclass(frozen=True, slots=True)
class DownloadedArtifact:
    media_type: str
    size: int
    sha256: str
    path: Path


type JsonErrorFactory = Callable[[JsonObject, int | None, str | None], ClientHttpError | None]


def _json_object(value: object, *, subject: str) -> JsonObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ClientProtocolError(f"{subject} 必须是 JSON object")
    return value


def _optional_json_object(response: httpx.Response) -> JsonObject | None:
    """尽力解析响应体：空体、非 JSON、非 object 都返回 None 而不抛协议错误。"""

    if not response.content:
        return None
    try:
        value: object = response.json()
    except ValueError:
        return None
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return value
    return None


def _parse_json_response(response: httpx.Response, *, subject: str) -> JsonObject:
    try:
        value: object = response.json()
    except ValueError as exc:
        raise ClientProtocolError(
            f"{subject} 返回了无效 JSON",
            status_code=response.status_code,
        ) from exc
    return _json_object(value, subject=subject)


_UPSTREAM_DETAIL_MAX_CHARS = 300


def truncate_upstream_detail(detail: str) -> str:
    """截断上游错误详情，避免故障/恶意上游膨胀用户可见错误与日志。"""

    if len(detail) > _UPSTREAM_DETAIL_MAX_CHARS:
        return f"{detail[:_UPSTREAM_DETAIL_MAX_CHARS]}…"
    return detail


def _detail_text(value: JsonValue) -> str | None:
    """把 code/message 槽位的值转成可展示文本；bool 是 int 子类，显式排除。"""

    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _fields_detail(source: Mapping[str, JsonValue]) -> list[str]:
    parts: list[str] = []
    for field in ("code", "message"):
        text = _detail_text(source.get(field))
        if text is not None and text not in parts:
            parts.append(text)
    return parts


def _upstream_error_detail(payload: JsonObject | None) -> str | None:
    """从错误响应体中尽力提取上游给出的失败原因。

    兜底 ClientHttpError 的文案会原样进入使用者可见的错误链路；上游把拒绝原因写在
    error.code/error.message（或顶层 code/message）里，也有网关直接用字符串 error
    或数字 code，丢掉它们就把「哪个参数被拒」压缩成一个裸 HTTP 400，没法排查。
    这里只做只读提取加截断，不做本地化——clients 层自包含，不引 core/i18n。
    """

    if payload is None:
        return None
    parts: list[str] = []
    error = payload.get("error")
    if isinstance(error, dict):
        parts = _fields_detail(error)
    else:
        direct = _detail_text(error)
        if direct is not None:
            parts = [direct]
    if not parts:
        parts = _fields_detail(payload)
    if not parts:
        return None
    detail = ": ".join(parts)
    return truncate_upstream_detail(detail)


def _with_upstream_detail(message: str, payload: JsonObject | None) -> str:
    detail = _upstream_error_detail(payload)
    return message if detail is None else f"{message}: {detail}"


def _merge_headers(base: Mapping[str, str], extra: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(base)
    if extra is not None:
        result.update(extra)
    return result


def encode_query_value(value: object) -> str:
    """按 HTTP query 既有语义序列化单个值。"""

    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _merge_query(base: Mapping[str, str], extra: Mapping[str, object] | None) -> dict[str, str]:
    result = dict(base)
    if extra is not None:
        for key, value in extra.items():
            if value is not None:
                result[key] = encode_query_value(value)
    return result


def _should_retry_response(response: httpx.Response, policy: RetryPolicy) -> bool:
    override = response.headers.get("x-should-retry")
    if override is not None:
        normalized = override.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return policy.should_retry_status(response.status_code)


def _absolute_url(base_url: str, path: str) -> str:
    parsed = urlsplit(path)
    if parsed.scheme:
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("资源 URL 必须是有效的 HTTP 或 HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("资源 URL 不允许包含用户凭据")
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _timeout(connection: HttpConnection) -> httpx.Timeout | None:
    if connection.request_timeout is None and connection.connect_timeout is None:
        return None
    request_timeout = connection.request_timeout
    if request_timeout is None:
        request_timeout = connection.connect_timeout
    return httpx.Timeout(request_timeout, connect=connection.connect_timeout)


def _parse_sse_payload(data: str, *, subject: str) -> JsonObject | None:
    normalized = data.strip()
    if not normalized or normalized == "[DONE]":
        return None
    try:
        value: object = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ClientProtocolError(f"{subject} SSE data 不是有效 JSON") from exc
    return _json_object(value, subject=f"{subject} SSE data")


class HttpSession:
    """绑定不可变 Connection 的轻量请求视图。"""

    def __init__(self, client: httpx.AsyncClient, connection: HttpConnection) -> None:
        self._client = client
        self.connection = connection

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        subject: str,
        json_body: Mapping[str, JsonValue] | None = None,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, object] | None = None,
        retry: RetryPolicy = NO_RETRY,
        error_factory: JsonErrorFactory | None = None,
    ) -> JsonObject:
        retries = max(retry.max_retries, 0)
        for attempt in range(retries + 1):
            try:
                response = await self._client.request(
                    method,
                    _absolute_url(self.connection.base_url, path),
                    json=dict(json_body) if json_body is not None else None,
                    headers=_merge_headers(self.connection.headers(), headers),
                    params=_merge_query(self.connection.query(), query),
                    timeout=_timeout(self.connection),
                )
            except httpx.TimeoutException as exc:
                if retry.retry_timeouts and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientTimeoutError(
                    f"{subject} 请求超时",
                    retryable=retry.retry_timeouts,
                    uncertain=retry.uncertain_on_timeout,
                ) from exc
            except httpx.HTTPError as exc:
                if retry.retry_connections and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientConnectionError(
                    f"{subject} 连接失败",
                    retryable=retry.retry_connections,
                ) from exc
            if response.status_code >= 400:
                if attempt < retries and _should_retry_response(response, retry):
                    await response.aclose()
                    await self._delay(retry.retry_interval)
                    continue
                # 错误路径的响应体尽力解析：网关/CDN 的错误页往往不是 JSON，
                # 之前先严格解析再判状态，协议错误会盖掉真正的 HTTP 失败，还把
                # 本该重试的 5xx 挡在重试判断之前。
                error_payload = _optional_json_object(response)
                if error_factory is not None and error_payload is not None:
                    parsed_error = error_factory(error_payload, response.status_code, None)
                    if parsed_error is not None:
                        raise parsed_error
                raise ClientHttpError(
                    _with_upstream_detail(f"{subject} 请求失败: HTTP {response.status_code}", error_payload),
                    status_code=response.status_code,
                    retryable=_should_retry_response(response, retry),
                )
            payload = _parse_json_response(response, subject=subject)
            if error_factory is not None:
                parsed_error = error_factory(payload, response.status_code, None)
                if parsed_error is not None:
                    raise parsed_error
            return payload
        raise AssertionError("JSON 请求重试循环异常退出")

    async def request_optional_json(
        self,
        method: str,
        path: str,
        *,
        subject: str,
        json_body: Mapping[str, JsonValue] | None = None,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, object] | None = None,
        retry: RetryPolicy = NO_RETRY,
        error_factory: JsonErrorFactory | None = None,
    ) -> JsonObject | None:
        """给"可能没有响应体"的端点用的 JSON 请求。

        `request_json` 在读状态码之前就要求响应体是 JSON object，对文档明确写着
        "本接口无返回参数"的端点（例如 ARK 的删除视频生成任务）来说，空体是正常的成功
        响应，按协议错误处理是错的。这里空体或非 object 体一律得到 None；出错时如果
        恰好带了可解析的错误体，仍然交给 error_factory，否则退回状态码错误。
        """

        retries = max(retry.max_retries, 0)
        for attempt in range(retries + 1):
            try:
                response = await self._client.request(
                    method,
                    _absolute_url(self.connection.base_url, path),
                    json=dict(json_body) if json_body is not None else None,
                    headers=_merge_headers(self.connection.headers(), headers),
                    params=_merge_query(self.connection.query(), query),
                    timeout=_timeout(self.connection),
                )
            except httpx.TimeoutException as exc:
                if retry.retry_timeouts and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientTimeoutError(
                    f"{subject} 请求超时",
                    retryable=retry.retry_timeouts,
                    uncertain=retry.uncertain_on_timeout,
                ) from exc
            except httpx.HTTPError as exc:
                if retry.retry_connections and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientConnectionError(
                    f"{subject} 连接失败",
                    retryable=retry.retry_connections,
                ) from exc
            payload = _optional_json_object(response)
            if response.status_code >= 400:
                if attempt < retries and _should_retry_response(response, retry):
                    await response.aclose()
                    await self._delay(retry.retry_interval)
                    continue
                if error_factory is not None and payload is not None:
                    parsed_error = error_factory(payload, response.status_code, None)
                    if parsed_error is not None:
                        raise parsed_error
                raise ClientHttpError(
                    _with_upstream_detail(f"{subject} 请求失败: HTTP {response.status_code}", payload),
                    status_code=response.status_code,
                    retryable=_should_retry_response(response, retry),
                )
            if error_factory is not None and payload is not None:
                parsed_error = error_factory(payload, response.status_code, None)
                if parsed_error is not None:
                    raise parsed_error
            return payload
        raise AssertionError("可选 JSON 请求重试循环异常退出")

    async def request_multipart(
        self,
        path: str,
        *,
        subject: str,
        form_data: Mapping[str, str],
        files: Mapping[str, tuple[str, bytes]],
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, object] | None = None,
        retry: RetryPolicy = NO_RETRY,
    ) -> httpx.Response:
        request_headers = _merge_headers(self.connection.headers(), headers)
        for key in list(request_headers):
            if key.lower() == "content-type":
                request_headers.pop(key)
        retries = max(retry.max_retries, 0)
        for attempt in range(retries + 1):
            try:
                response = await self._client.post(
                    _absolute_url(self.connection.base_url, path),
                    data=dict(form_data),
                    files=dict(files),
                    headers=request_headers,
                    params=_merge_query(self.connection.query(), query),
                    timeout=_timeout(self.connection),
                )
            except httpx.TimeoutException as exc:
                if retry.retry_timeouts and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientTimeoutError(f"{subject} 请求超时", retryable=retry.retry_timeouts) from exc
            except httpx.HTTPError as exc:
                if retry.retry_connections and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientConnectionError(f"{subject} 连接失败", retryable=retry.retry_connections) from exc
            if response.status_code >= 400 and attempt < retries and _should_retry_response(response, retry):
                await response.aclose()
                await self._delay(retry.retry_interval)
                continue
            if response.status_code >= 400:
                raise ClientHttpError(
                    _with_upstream_detail(
                        f"{subject} 请求失败: HTTP {response.status_code}", _optional_json_object(response)
                    ),
                    status_code=response.status_code,
                    retryable=_should_retry_response(response, retry),
                )
            return response
        raise AssertionError("multipart 请求重试循环异常退出")

    async def stream_sse_json(
        self,
        path: str,
        *,
        subject: str,
        json_body: Mapping[str, JsonValue],
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, object] | None = None,
        retry: RetryPolicy = NO_RETRY,
        error_factory: JsonErrorFactory | None = None,
    ) -> AsyncIterator[SseJsonEvent]:
        retries = max(retry.max_retries, 0)
        for attempt in range(retries + 1):
            emitted_event = False
            try:
                async with self._client.stream(
                    "POST",
                    _absolute_url(self.connection.base_url, path),
                    json=dict(json_body),
                    headers=_merge_headers(self.connection.headers(), headers),
                    params=_merge_query(self.connection.query(), query),
                    timeout=_timeout(self.connection),
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        if attempt < retries and _should_retry_response(response, retry):
                            await self._delay(retry.retry_interval)
                            continue
                        error_payload = _optional_json_object(response)
                        if error_factory is not None and error_payload is not None:
                            parsed_error = error_factory(error_payload, response.status_code, None)
                            if parsed_error is not None:
                                raise parsed_error
                        raise ClientHttpError(
                            _with_upstream_detail(
                                f"{subject} SSE 请求失败: HTTP {response.status_code}", error_payload
                            ),
                            status_code=response.status_code,
                        )
                    event_name: str | None = None
                    event_status: int | None = None
                    data_lines: list[str] = []
                    async for raw_line in response.aiter_lines():
                        line = raw_line.rstrip("\r")
                        if not line:
                            payload = _parse_sse_payload("\n".join(data_lines), subject=subject)
                            if payload is not None:
                                if error_factory is not None:
                                    parsed_error = error_factory(payload, event_status, event_name)
                                    if parsed_error is not None:
                                        raise parsed_error
                                emitted_event = True
                                yield SseJsonEvent(event=event_name, data=payload, status=event_status)
                            elif data_lines and "\n".join(data_lines).strip() == "[DONE]":
                                return
                            event_name = None
                            event_status = None
                            data_lines.clear()
                            continue
                        if line.startswith(":"):
                            continue
                        if line.startswith("event:"):
                            event_name = line.removeprefix("event:").lstrip()
                        elif line.startswith("status:"):
                            raw_status = line.removeprefix("status:").strip()
                            event_status = int(raw_status) if raw_status.isdigit() else None
                        elif line.startswith("data:"):
                            data_lines.append(line.removeprefix("data:").lstrip())
                    if data_lines:
                        payload = _parse_sse_payload("\n".join(data_lines), subject=subject)
                        if payload is not None:
                            if error_factory is not None:
                                parsed_error = error_factory(payload, event_status, event_name)
                                if parsed_error is not None:
                                    raise parsed_error
                            emitted_event = True
                            yield SseJsonEvent(event=event_name, data=payload, status=event_status)
                    return
            except httpx.TimeoutException as exc:
                if not emitted_event and retry.retry_timeouts and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientTimeoutError(
                    f"{subject} SSE 请求超时",
                    retryable=not emitted_event and retry.retry_timeouts,
                    uncertain=retry.uncertain_on_timeout,
                ) from exc
            except httpx.HTTPError as exc:
                if not emitted_event and retry.retry_connections and attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientConnectionError(
                    f"{subject} SSE 连接失败",
                    retryable=not emitted_event and retry.retry_connections,
                ) from exc
        raise AssertionError("SSE 请求重试循环异常退出")

    async def upload_file(
        self,
        url: str,
        *,
        subject: str,
        form_data: Mapping[str, str],
        file_field: str,
        file_path: Path,
        media_type: str,
        retry: RetryPolicy,
    ) -> None:
        retries = max(retry.max_retries, 0)
        for attempt in range(retries + 1):
            try:
                with file_path.open("rb") as stream:
                    response = await self._client.post(
                        _absolute_url(self.connection.base_url, url),
                        data=dict(form_data),
                        files={file_field: (file_path.name, stream, media_type)},
                        timeout=_timeout(self.connection),
                    )
            except httpx.HTTPError as exc:
                if attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientConnectionError(f"{subject} 上传失败", retryable=True) from exc
            if response.is_success:
                return
            if attempt < retries and _should_retry_response(response, retry):
                await self._delay(retry.retry_interval)
                continue
            raise ClientHttpError(
                _with_upstream_detail(
                    f"{subject} 上传失败: HTTP {response.status_code}", _optional_json_object(response)
                ),
                status_code=response.status_code,
                retryable=_should_retry_response(response, retry),
            )
        raise AssertionError("文件上传重试循环异常退出")

    async def download(
        self,
        url: str,
        destination: Path,
        *,
        subject: str,
        max_bytes: int,
        retry: RetryPolicy,
    ) -> DownloadedArtifact:
        if urlsplit(url).scheme.lower() != "https":
            raise ValueError("artifact URL 必须使用 HTTPS")
        staging = destination.with_name(f".{destination.name}.part")
        retries = max(retry.max_retries, 0)
        for attempt in range(retries + 1):
            staging.unlink(missing_ok=True)
            try:
                async with self._client.stream(
                    "GET",
                    url,
                    follow_redirects=True,
                    timeout=_timeout(self.connection),
                ) as response:
                    if not response.is_success:
                        if attempt < retries and _should_retry_response(response, retry):
                            await self._delay(retry.retry_interval)
                            continue
                        raise ClientHttpError(
                            f"{subject} 下载失败: HTTP {response.status_code}",
                            status_code=response.status_code,
                            retryable=_should_retry_response(response, retry),
                        )
                    digest = sha256()
                    size = 0
                    with staging.open("wb") as output:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise ClientProtocolError(f"{subject} 超过最大字节数 {max_bytes}")
                            output.write(chunk)
                            digest.update(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    staging.replace(destination)
                    return DownloadedArtifact(
                        media_type=response.headers.get("content-type", "application/octet-stream").split(";", 1)[0],
                        size=size,
                        sha256=digest.hexdigest(),
                        path=destination,
                    )
            except ClientHttpError:
                staging.unlink(missing_ok=True)
                raise
            except httpx.HTTPError as exc:
                staging.unlink(missing_ok=True)
                if attempt < retries:
                    await self._delay(retry.retry_interval)
                    continue
                raise ClientConnectionError(f"{subject} 下载失败", retryable=True) from exc
        raise AssertionError("文件下载重试循环异常退出")

    @staticmethod
    async def _delay(seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)


class SharedHttpClient:
    """供应商运行时共享的连接池与 Session 租约管理器。"""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(transport=transport)
        self._condition = asyncio.Condition()
        self._active_sessions = 0
        self._closing = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_sessions(self) -> int:
        return self._active_sessions

    @asynccontextmanager
    async def session(self, connection: HttpConnection) -> AsyncIterator[HttpSession]:
        async with self._condition:
            if self._closing or self._closed:
                raise ClientClosedError("Client 已关闭")
            self._active_sessions += 1
        try:
            yield HttpSession(self._client, connection)
        finally:
            async with self._condition:
                self._active_sessions -= 1
                if self._active_sessions == 0:
                    self._condition.notify_all()

    async def aclose(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closing = True
            await self._condition.wait_for(lambda: self._active_sessions == 0)
            self._closed = True
        await self._client.aclose()
