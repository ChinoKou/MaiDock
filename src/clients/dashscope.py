from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from .common import (
    ClientHttpError,
    DownloadedArtifact,
    HttpConnection,
    HttpSession,
    JsonObject,
    RetryPolicy,
    SharedHttpClient,
    SseJsonEvent,
    truncate_upstream_detail,
)
from .families import JsonResource, JsonResourceRequest


class DashScopeClientError(ClientHttpError):
    """DashScope 原生错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        upstream_message: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        uncertain: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            retryable=retryable,
            uncertain=uncertain,
        )
        self.code = code
        self.upstream_message = upstream_message or message
        self.request_id = request_id

    @property
    def is_endpoint_mismatch(self) -> bool:
        return "invalidparameter" in self.code.lower() and "url error" in self.upstream_message.lower()


def dashscope_error_factory(
    payload: JsonObject,
    status_code: int | None,
    event_name: str | None,
) -> DashScopeClientError | None:
    raw_code = payload.get("code")
    raw_message = payload.get("message")
    payload_status = payload.get("status_code")
    effective_status = status_code if status_code is not None else payload_status
    if not isinstance(effective_status, int):
        effective_status = None
    normalized_code = str(raw_code).strip() if raw_code is not None else ""
    has_error_code = bool(normalized_code) and normalized_code.lower() not in {"success", "ok"}
    has_error_status = effective_status is not None and not 200 <= effective_status < 300
    if event_name != "error" and not has_error_code and not has_error_status:
        return None
    code = normalized_code or "UPSTREAM_ERROR"
    request_id = payload.get("request_id") or payload.get("requestId")
    normalized_request_id = request_id if isinstance(request_id, str) else None
    upstream_message = truncate_upstream_detail(str(raw_message or "上游未提供错误消息"))
    details = [f"code={code}", f"message={upstream_message}"]
    if effective_status is not None:
        details.insert(0, f"status={effective_status}")
    if normalized_request_id is not None:
        details.append(f"request_id={normalized_request_id}")
    message = f"DashScope 请求失败: {' '.join(details)}"
    return DashScopeClientError(
        code,
        message,
        upstream_message=upstream_message,
        status_code=effective_status,
        retryable=(effective_status in {408, 409, 429} or effective_status is not None and effective_status >= 500),
        request_id=normalized_request_id,
    )


def bailian_responses_error_factory(
    payload: JsonObject,
    status_code: int | None,
    event_name: str | None,
) -> DashScopeClientError | None:
    """兼容百炼 Responses 的 OpenAI 嵌套与 DashScope 顶层错误。"""

    raw_error = payload.get("error")
    error_mapping = raw_error if isinstance(raw_error, dict) else None
    if error_mapping is None:
        dashscope_error = dashscope_error_factory(payload, status_code, event_name)
        if dashscope_error is None:
            return None
        return DashScopeClientError(
            dashscope_error.code,
            str(dashscope_error).replace("DashScope 请求失败:", "百炼 Responses 请求失败:", 1),
            upstream_message=dashscope_error.upstream_message,
            status_code=dashscope_error.status_code,
            retryable=dashscope_error.retryable,
            uncertain=dashscope_error.uncertain,
            request_id=dashscope_error.request_id,
        )
    code = str(error_mapping.get("code") or "").strip() if error_mapping is not None else ""
    message = str(error_mapping.get("message") or "").strip() if error_mapping is not None else ""
    error_type = str(error_mapping.get("type") or "").strip() if error_mapping is not None else ""
    has_error = bool(code or message or error_type)
    has_error_status = status_code is not None and not 200 <= status_code < 300
    if event_name != "error" and not has_error and not has_error_status:
        return None
    normalized_code = code or error_type or "UPSTREAM_ERROR"
    request_id = payload.get("request_id") or payload.get("requestId")
    normalized_request_id = request_id if isinstance(request_id, str) else None
    upstream_message = truncate_upstream_detail(message or "上游未提供错误消息")
    details = [f"code={normalized_code}", f"message={upstream_message}"]
    if status_code is not None:
        details.insert(0, f"status={status_code}")
    if normalized_request_id is not None:
        details.append(f"request_id={normalized_request_id}")
    return DashScopeClientError(
        normalized_code,
        f"百炼 Responses 请求失败: {' '.join(details)}",
        upstream_message=upstream_message,
        status_code=status_code,
        retryable=(status_code in {408, 409, 429} or status_code is not None and status_code >= 500),
        request_id=normalized_request_id,
    )


@dataclass(frozen=True, slots=True)
class DashScopePaths:
    text_generation: str
    multimodal_generation: str
    embeddings: str
    image_generation: str
    text2image_synthesis: str
    image2image_synthesis: str
    video_generation: str
    tasks: str = "tasks"
    uploads: str = "uploads"


@dataclass(frozen=True, slots=True)
class DashScopeConnection:
    http: HttpConnection
    retry: RetryPolicy
    safe_retry: RetryPolicy
    paths: DashScopePaths


@dataclass(frozen=True, slots=True)
class DashScopeResponsesConnection:
    """百炼 Responses 专属连接，不携带 DashScope 原生资源路径。"""

    http: HttpConnection
    retry: RetryPolicy
    responses_path: str


class DashScopeResponsesResource:
    """百炼 Responses API 资源（endpoint 为 base_url 追加 /responses）。"""

    def __init__(self, session: HttpSession, *, path: str) -> None:
        self._resource = JsonResource(
            session,
            path=path,
            subject="Bailian Responses",
            error_factory=bailian_responses_error_factory,
        )

    async def create(self, request: JsonResourceRequest, *, retry: RetryPolicy) -> JsonObject:
        return await self._resource.create(request, retry=retry)

    def stream(self, request: JsonResourceRequest, *, retry: RetryPolicy) -> AsyncIterator[SseJsonEvent]:
        return self._resource.stream(request, retry=retry)


class DashScopeResponsesSession:
    """百炼 Responses 专属 Session：只暴露 responses 资源。"""

    def __init__(self, session: HttpSession, connection: DashScopeResponsesConnection) -> None:
        self.retry = connection.retry
        self.responses = DashScopeResponsesResource(session, path=connection.responses_path)


class DashScopeTasksResource:
    def __init__(self, session: HttpSession, *, base_path: str) -> None:
        self._session = session
        self._base_path = base_path.rstrip("/")

    async def retrieve(self, task_id: str, *, retry: RetryPolicy) -> JsonObject:
        return await self._session.request_json(
            "GET",
            f"{self._base_path}/{task_id}",
            subject="DashScope Tasks Retrieve",
            retry=retry,
            error_factory=dashscope_error_factory,
        )

    async def cancel(self, task_id: str, *, retry: RetryPolicy) -> JsonObject:
        return await self._session.request_json(
            "POST",
            f"{self._base_path}/{task_id}/cancel",
            subject="DashScope Tasks Cancel",
            retry=retry,
            error_factory=dashscope_error_factory,
        )


class DashScopeUploadsResource:
    def __init__(self, session: HttpSession, *, path: str) -> None:
        self._session = session
        self._path = path

    async def create_policy(self, model: str, *, retry: RetryPolicy) -> JsonObject:
        return await self._session.request_json(
            "GET",
            self._path,
            subject="DashScope Upload Policy",
            query={"action": "getPolicy", "model": model},
            retry=retry,
            error_factory=dashscope_error_factory,
        )

    async def upload(
        self,
        *,
        upload_url: str,
        form_data: Mapping[str, str],
        path: Path,
        media_type: str,
        retry: RetryPolicy,
    ) -> None:
        await self._session.upload_file(
            upload_url,
            subject="DashScope OSS",
            form_data=form_data,
            file_field="file",
            file_path=path,
            media_type=media_type,
            retry=retry,
        )


class DashScopeArtifactsResource:
    def __init__(self, session: HttpSession) -> None:
        self._session = session

    async def download(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        retry: RetryPolicy,
    ) -> DownloadedArtifact:
        return await self._session.download(
            url,
            destination,
            subject="DashScope Artifact",
            max_bytes=max_bytes,
            retry=retry,
        )


class DashScopeSession:
    def __init__(self, session: HttpSession, connection: DashScopeConnection) -> None:
        self.retry = connection.retry
        self.safe_retry = connection.safe_retry
        paths = connection.paths
        self.text_generation = JsonResource(
            session,
            path=paths.text_generation,
            subject="DashScope Text Generation",
            error_factory=dashscope_error_factory,
        )
        self.multimodal_generation = JsonResource(
            session,
            path=paths.multimodal_generation,
            subject="DashScope Multimodal Generation",
            error_factory=dashscope_error_factory,
        )
        self.embeddings = JsonResource(
            session,
            path=paths.embeddings,
            subject="DashScope Embeddings",
            error_factory=dashscope_error_factory,
        )
        self.audio_transcriptions = self.multimodal_generation
        self.image_generation = JsonResource(
            session,
            path=paths.image_generation,
            subject="DashScope Image Generation",
            error_factory=dashscope_error_factory,
        )
        self.text2image_synthesis = JsonResource(
            session,
            path=paths.text2image_synthesis,
            subject="DashScope Text2Image Synthesis",
            error_factory=dashscope_error_factory,
        )
        self.image2image_synthesis = JsonResource(
            session,
            path=paths.image2image_synthesis,
            subject="DashScope Image2Image Synthesis",
            error_factory=dashscope_error_factory,
        )
        self.video_generation = JsonResource(
            session,
            path=paths.video_generation,
            subject="DashScope Video Generation",
            error_factory=dashscope_error_factory,
        )
        self.tasks = DashScopeTasksResource(session, base_path=paths.tasks)
        self.uploads = DashScopeUploadsResource(session, path=paths.uploads)
        self.artifacts = DashScopeArtifactsResource(session)


class DashScopeClient:
    """DashScope 原生协议资源 Client。"""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = SharedHttpClient(transport=transport)

    @property
    def closed(self) -> bool:
        return self._http.closed

    @asynccontextmanager
    async def session(self, connection: DashScopeConnection) -> AsyncIterator[DashScopeSession]:
        async with self._http.session(connection.http) as session:
            yield DashScopeSession(session, connection)

    @asynccontextmanager
    async def responses_session(
        self,
        connection: DashScopeResponsesConnection,
    ) -> AsyncIterator[DashScopeResponsesSession]:
        """百炼 Responses 专属 Session，与原生通路共享同一个连接池。"""

        async with self._http.session(connection.http) as session:
            yield DashScopeResponsesSession(session, connection)

    async def aclose(self) -> None:
        await self._http.aclose()
