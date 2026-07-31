from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field

import httpx

from ..common import HttpSession, JsonErrorFactory, JsonObject, JsonValue, RetryPolicy, SseJsonEvent


@dataclass(frozen=True, slots=True)
class JsonResourceRequest:
    """协议资源已经形成的 wire 请求。"""

    body: Mapping[str, JsonValue]
    headers: Mapping[str, str] = field(default_factory=dict)
    query: Mapping[str, object] = field(default_factory=dict)


class JsonResource:
    """绑定固定端点和错误解析器的 JSON 协议资源。"""

    def __init__(
        self,
        session: HttpSession,
        *,
        path: str,
        subject: str,
        error_factory: JsonErrorFactory | None = None,
    ) -> None:
        self._session = session
        self._path = path
        self._subject = subject
        self._error_factory = error_factory

    async def create(
        self,
        request: JsonResourceRequest,
        *,
        retry: RetryPolicy,
    ) -> JsonObject:
        return await self._session.request_json(
            "POST",
            self._path,
            subject=self._subject,
            json_body=request.body,
            headers=request.headers,
            query=request.query,
            retry=retry,
            error_factory=self._error_factory,
        )

    def stream(
        self,
        request: JsonResourceRequest,
        *,
        retry: RetryPolicy,
    ) -> AsyncIterator[SseJsonEvent]:
        return self._session.stream_sse_json(
            self._path,
            subject=self._subject,
            json_body=request.body,
            headers=request.headers,
            query=request.query,
            retry=retry,
            error_factory=self._error_factory,
        )


@dataclass(frozen=True, slots=True)
class MultipartResourceRequest:
    """协议资源已经形成的 multipart 请求。"""

    form_data: Mapping[str, str]
    files: Mapping[str, tuple[str, bytes]]
    headers: Mapping[str, str] = field(default_factory=dict)
    query: Mapping[str, object] = field(default_factory=dict)


class MultipartResource:
    """绑定固定端点的 multipart 协议资源。"""

    def __init__(self, session: HttpSession, *, path: str, subject: str) -> None:
        self._session = session
        self._path = path
        self._subject = subject

    async def create(
        self,
        request: MultipartResourceRequest,
        *,
        retry: RetryPolicy,
    ) -> httpx.Response:
        return await self._session.request_multipart(
            self._path,
            subject=self._subject,
            form_data=request.form_data,
            files=request.files,
            headers=request.headers,
            query=request.query,
            retry=retry,
        )
