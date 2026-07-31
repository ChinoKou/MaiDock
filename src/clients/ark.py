from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from .common import (
    NO_RETRY,
    ClientHttpError,
    DownloadedArtifact,
    HttpConnection,
    HttpSession,
    JsonObject,
    RetryPolicy,
    SharedHttpClient,
)
from .families import JsonResource

ARK_IMAGES_GENERATIONS_PATH = "images/generations"
ARK_CONTENT_GENERATION_TASKS_PATH = "contents/generations/tasks"


class ArkClientError(ClientHttpError):
    """Volcengine ARK 原生错误。"""

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


def ark_error_factory(
    payload: JsonObject,
    status_code: int | None,
    event_name: str | None,
) -> ArkClientError | None:
    """把 ARK 的错误响应体转成 ArkClientError。

    只认顶层 `error` 对象：ARK 把整单失败放在 `{"error": {"code", "message"}}`，
    2xx 响应里出现它同样意味着整单失败，所以这里不看状态码就先判。图片接口
    `data[]` 里的逐项 error 是"部分成功"，不属于整单失败，留给 driver 转成 warnings。
    """

    error = payload.get("error")
    error_code: str | None = None
    error_message: str | None = None
    if isinstance(error, dict):
        raw_code = error.get("code")
        raw_message = error.get("message")
        error_code = str(raw_code).strip() if raw_code is not None else None
        error_message = str(raw_message) if raw_message is not None else None

    has_error = bool(error_code) or bool(error_message)
    has_error_status = status_code is not None and not 200 <= status_code < 300
    if event_name != "error" and not has_error and not has_error_status:
        return None

    code = error_code or "UPSTREAM_ERROR"
    upstream_message = error_message or "上游未提供错误消息"
    raw_request_id = payload.get("request_id") or payload.get("id")
    request_id = raw_request_id if isinstance(raw_request_id, str) else None

    details = [f"code={code}", f"message={upstream_message}"]
    if status_code is not None:
        details.insert(0, f"status={status_code}")
    if request_id is not None:
        details.append(f"request_id={request_id}")
    return ArkClientError(
        code,
        f"Volcengine ARK 请求失败: {' '.join(details)}",
        upstream_message=upstream_message,
        status_code=status_code,
        retryable=(status_code in {408, 409, 429} or status_code is not None and status_code >= 500),
        request_id=request_id,
    )


@dataclass(frozen=True, slots=True)
class ArkConnection:
    http: HttpConnection
    retry: RetryPolicy
    responses_path: str
    embeddings_path: str
    audio_transcriptions_path: str
    tokenization_path: str
    # 以下三项只有公共媒体通路会用到，给默认值让既有 Host 通路的构造点保持不变。
    images_generations_path: str = ARK_IMAGES_GENERATIONS_PATH
    content_generation_tasks_path: str = ARK_CONTENT_GENERATION_TASKS_PATH
    # 幂等操作（查任务、下载产物）专用的重试策略，与提交用的 retry 分开：
    # 提交必须零重试以免重复计费，查询和下载重试是安全的。
    safe_retry: RetryPolicy = NO_RETRY


class ArkContentGenerationTasksResource:
    """ARK 视频生成异步任务资源。"""

    def __init__(self, session: HttpSession, *, base_path: str) -> None:
        self._session = session
        self._base_path = base_path.rstrip("/")

    async def create(self, body: JsonObject, *, retry: RetryPolicy) -> JsonObject:
        return await self._session.request_json(
            "POST",
            self._base_path,
            subject="Volcengine Ark Content Generation Task Create",
            json_body=body,
            retry=retry,
            error_factory=ark_error_factory,
        )

    async def retrieve(self, task_id: str, *, retry: RetryPolicy) -> JsonObject:
        return await self._session.request_json(
            "GET",
            f"{self._base_path}/{task_id}",
            subject="Volcengine Ark Content Generation Task Retrieve",
            retry=retry,
            error_factory=ark_error_factory,
        )

    async def delete(self, task_id: str, *, retry: RetryPolicy) -> JsonObject | None:
        """删除或取消任务。文档原文"本接口无返回参数"，因此正常成功时返回 None。

        ARK 用同一个 DELETE 表达两件事，取决于任务当时的状态：

        | 状态 | 支持 DELETE | 含义 |
        | --- | --- | --- |
        | queued | 是 | 取消排队，状态变为 cancelled |
        | running | 否 | — |
        | succeeded / failed / expired | 是 | **删除任务记录，此后无法查询** |
        | cancelled | 否 | — |

        也就是说只有 queued 才是"取消"，终态上调用等于不可逆地销毁记录。调用方必须
        先确认状态，绝不能把它当成通用的 cancel。资源层只如实暴露这个动词，语义判断
        留在 driver 的状态机里。
        """

        return await self._session.request_optional_json(
            "DELETE",
            f"{self._base_path}/{task_id}",
            subject="Volcengine Ark Content Generation Task Delete",
            retry=retry,
            error_factory=ark_error_factory,
        )


class ArkArtifactsResource:
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
            subject="Volcengine Ark Artifact",
            max_bytes=max_bytes,
            retry=retry,
        )


class ArkSession:
    def __init__(self, session: HttpSession, connection: ArkConnection) -> None:
        self.retry = connection.retry
        self.safe_retry = connection.safe_retry
        # 既有四个文本资源保持不带 error_factory：Host 通路的错误文案由 host_adapters
        # 自己映射，这里换成 ArkClientError 会改动已有断言。媒体资源是新链路，直接用工厂。
        self.responses = JsonResource(session, path=connection.responses_path, subject="Volcengine Ark Responses")
        self.embeddings = JsonResource(session, path=connection.embeddings_path, subject="Volcengine Ark Embeddings")
        self.audio_transcriptions = JsonResource(
            session,
            path=connection.audio_transcriptions_path,
            subject="Volcengine Ark Audio Transcriptions",
        )
        self.tokenization = JsonResource(
            session,
            path=connection.tokenization_path,
            subject="Volcengine Ark Tokenization",
        )
        self.images_generations = JsonResource(
            session,
            path=connection.images_generations_path,
            subject="Volcengine Ark Images Generations",
            error_factory=ark_error_factory,
        )
        self.content_generation_tasks = ArkContentGenerationTasksResource(
            session,
            base_path=connection.content_generation_tasks_path,
        )
        self.artifacts = ArkArtifactsResource(session)


class ArkClient:
    """Volcengine Ark 原生资源 Client。"""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = SharedHttpClient(transport=transport)

    @property
    def closed(self) -> bool:
        return self._http.closed

    @asynccontextmanager
    async def session(self, connection: ArkConnection) -> AsyncIterator[ArkSession]:
        async with self._http.session(connection.http) as session:
            yield ArkSession(session, connection)

    async def aclose(self) -> None:
        await self._http.aclose()
